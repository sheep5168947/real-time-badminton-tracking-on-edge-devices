# -*- coding: utf-8 -*-
"""
Jetson Nano Real-Time Streaming Hardware Ablation Benchmark
Evaluates tracking performance under STRICT 30 FPS real-time camera ingestion.
Measures:
1. Real-time effective FPS and latency
2. Hardware Frame Drop Count & Frame Drop Rate (%)
3. Real-time Precision, Recall, and F1-Score (accounting for dropped frames)
4. Local Search vs Global Recovery engine ratios

Variants supported:
1. native_global: Native TrackNetV4 (Pure Global 512x288, no ROI)
2. baseline: MiniROI 128x128 + Static ROI + Raw Heatmap
3. no_mask: MiniROI 128x128 + Kalman Filter + Raw Heatmap
4. no_kf: MiniROI 128x128 + Static ROI + Motion Mask
5. proposed: MiniROI 128x128 + Kalman Filter + Motion Mask
6. all: Runs each variant in an isolated OS subprocess to guarantee 0% memory leak / no OOM.
"""

import os
import sys

# Critical runtime guards for Jetson Nano (Cortex-A57 & OpenMP TLS)
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
if "/usr/lib/aarch64-linux-gnu/libgomp.so.1" not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1:" + os.environ.get("LD_PRELOAD", "")

import cv2
import time
import numpy as np
import threading
import argparse
import math
import csv
import subprocess
import gc
from collections import deque
import pycuda.driver as cuda
import tensorrt as trt

VARIANTS_MAP = {
    "native_global": {
        "name": "Native TrackNetV4 (Pure Global 512x288)",
        "use_kf": False,
        "use_mask": False,
        "pure_global": True
    },
    "baseline": {
        "name": "MiniROI Baseline (Static + Raw)",
        "use_kf": False,
        "use_mask": False,
        "pure_global": False
    },
    "no_mask": {
        "name": "MiniROI No Mask (KF + Raw Heatmap)",
        "use_kf": True,
        "use_mask": False,
        "pure_global": False
    },
    "no_kf": {
        "name": "MiniROI No KF (Static ROI + Mask)",
        "use_kf": False,
        "use_mask": True,
        "pure_global": False
    },
    "proposed": {
        "name": "MiniROI Proposed (KF + Motion Mask)",
        "use_kf": True,
        "use_mask": True,
        "pure_global": False
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano Real-Time Hardware Ablation")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "native_global", "baseline", "no_mask", "no_kf", "proposed"],
                        help="Variant to run, or 'all' to run sequentially in isolated subprocesses")
    parser.add_argument("--video", type=str, default="weights/1_01_00.mp4", help="Video file path")
    parser.add_argument("--csv", type=str, default="weights/gt.csv", help="Ground Truth CSV file path")
    parser.add_argument("--output_csv", type=str, default="report_materials/nano_realtime_summary.csv", help="Output summary CSV")
    parser.add_argument("--engine_128", type=str, default="weights/miniroi_128x128.engine")
    parser.add_argument("--engine_512", type=str, default="weights/miniroi_512x288.engine")
    parser.add_argument("--motion", type=int, default=10)
    parser.add_argument("--local_conf", type=float, default=0.55)
    parser.add_argument("--global_conf", type=float, default=0.55)
    parser.add_argument("--miss", type=int, default=3)
    return parser.parse_args()

class TRTModel:
    def __init__(self, engine_path, ctx):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def __call__(self, x_np):
        np.copyto(self.inputs[0]['host'], x_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy().reshape(self.outputs[0]['shape'])

def fast_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def get_shuttle_center_local(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0: return 0, 0, 0
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(idx[-1]), int(idx[-2]), float(max_val)

def crop_and_pad(img, center_x, center_y, size):
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = int(center_x - half), int(center_x + half)
    y_min, y_max = int(center_y - half), int(center_y + half)
    valid_x_min, valid_x_max = max(0, x_min), min(w, x_max)
    valid_y_min, valid_y_max = max(0, y_min), min(h, y_max)
    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, img.shape[2] if len(img.shape)==3 else 1), dtype=img.dtype)
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        padded[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
        return padded, x_min, y_min
    return cropped, x_min, y_min

def init_kalman_filter(init_x, init_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
    return kf

def run_realtime_trial(var_name, use_kf, use_mask, pure_global, args, GT_DICT, W_orig, H_orig):
    print("\n" + "="*80)
    print("[RUNNING REAL-TIME TRIAL] Variant: {}".format(var_name))
    print("Mode: {}, KF={}, Mask={}".format(
        "Pure Global (512x288)" if pure_global else "MiniROI (128x128)",
        "ON" if use_kf else "OFF",
        "ON" if use_mask else "OFF"
    ))
    print("="*80)

    W_TRAIN, H_TRAIN = 512, 288
    ROI_SIZE = 128
    DIST_THRESH = 15.0 * (float(W_orig) / 1280.0)
    scale_w = float(W_orig) / float(W_TRAIN)
    scale_h = float(H_orig) / float(H_TRAIN)

    # Thread shared state
    shared = {
        "frame_id": -1,
        "frame_data": None,
        "is_running": True,
        "ai_ready": False,
        "predictions": {},   # frame_id -> (pred_x_512, pred_y_512) or None
        "processed_ids": set()
    }
    frame_lock = threading.Lock()

    # --- AI Worker Thread ---
    def ai_worker():
        cuda.init()
        cuda_ctx = cuda.Device(0).make_context()
        trt_128 = None
        trt_512 = None
        try:
            if not pure_global:
                trt_128 = TRTModel(args.engine_128, cuda_ctx)
            trt_512 = TRTModel(args.engine_512, cuda_ctx)
        except Exception as e:
            print("[ERROR] TRT Init Failed: {}".format(e))
            shared["is_running"] = False
            cuda_ctx.pop()
            return

        # Warm-up
        if trt_128 is not None:
            _ = trt_128(np.zeros((1, 9, 128, 128), dtype=np.float32))
        _ = trt_512(np.zeros((1, 9, 288, 512), dtype=np.float32))

        shared["ai_ready"] = True
        ai_queue = deque(maxlen=3)
        last_id = -1

        is_lost = True
        miss_count = 0
        kf = None
        last_x_512, last_y_512 = W_TRAIN // 2, H_TRAIN // 2

        stats = {"local": 0, "global": 0, "ai_time": 0.0, "ai_frames": 0}

        while shared["is_running"]:
            with frame_lock:
                f_id = shared["frame_id"]
                f_data = shared["frame_data"]

            if f_id <= last_id or f_data is None:
                time.sleep(0.001)
                continue

            last_id = f_id
            t0 = time.perf_counter()

            f_train = cv2.resize(f_data, (W_TRAIN, H_TRAIN))
            ai_queue.append(f_train)

            if len(ai_queue) < 3:
                continue

            img0, img1, img2 = ai_queue[0], ai_queue[1], ai_queue[2]
            pred_point = None

            if pure_global:
                # Pure Global TrackNetV4 Mode (512x288 every frame)
                imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
                tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                raw_out = trt_512(tx)
                heatmap = fast_sigmoid(raw_out).squeeze()

                if use_mask:
                    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    _, m = cv2.threshold(cv2.absdiff(g1, g0), args.motion, 255, cv2.THRESH_BINARY)
                    heatmap = heatmap * cv2.resize(m.astype(np.float32)/255.0, (512, 288))

                gx, gy, conf = get_shuttle_center_local(heatmap)
                if conf > args.global_conf:
                    pred_point = (gx, gy)
                stats["global"] += 1
            else:
                # Dual-Engine Hybrid Mode (MiniROI 128x128 + Global Recovery)
                # Phase 1: Local ROI
                if not is_lost:
                    if use_kf and kf is not None:
                        p = kf.predict()
                        px, py = int(p[0, 0]), int(p[1, 0])
                    else:
                        px, py = int(last_x_512), int(last_y_512)

                    c0, _, _ = crop_and_pad(img0, px, py, ROI_SIZE)
                    c1, _, _ = crop_and_pad(img1, px, py, ROI_SIZE)
                    c2, _, _ = crop_and_pad(img2, px, py, ROI_SIZE)

                    imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
                    tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                    raw_out = trt_128(tx)
                    heatmap = fast_sigmoid(raw_out).squeeze()

                    if use_mask:
                        g0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                        g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                        _, m = cv2.threshold(cv2.absdiff(g1, g0), args.motion, 255, cv2.THRESH_BINARY)
                        heatmap = heatmap * cv2.resize(m.astype(np.float32)/255.0, (128, 128))

                    lx, ly, conf = get_shuttle_center_local(heatmap)
                    if conf >= args.local_conf:
                        gx = px - ROI_SIZE // 2 + lx
                        gy = py - ROI_SIZE // 2 + ly
                        if use_kf and kf is not None:
                            kf.correct(np.array([[gx], [gy]], dtype=np.float32))
                        last_x_512, last_y_512 = gx, gy
                        pred_point = (gx, gy)
                        miss_count = 0
                        stats["local"] += 1
                    else:
                        miss_count += 1
                        # If KF is enabled, use KF state prediction during miss
                        if use_kf and kf is not None:
                            pred_point = (px, py)
                            last_x_512, last_y_512 = px, py
                        if miss_count >= args.miss:
                            is_lost = True

                # Phase 2: Global Recovery
                if is_lost:
                    imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
                    tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                    raw_out = trt_512(tx)
                    heatmap = fast_sigmoid(raw_out).squeeze()

                    if use_mask:
                        g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                        _, m = cv2.threshold(cv2.absdiff(g1, g0), args.motion, 255, cv2.THRESH_BINARY)
                        heatmap = heatmap * cv2.resize(m.astype(np.float32)/255.0, (512, 288))

                    gx, gy, conf = get_shuttle_center_local(heatmap)
                    if conf > args.global_conf:
                        if use_kf:
                            kf = init_kalman_filter(gx, gy)
                        last_x_512, last_y_512 = gx, gy
                        pred_point = (gx, gy)
                        is_lost = False
                        miss_count = 0
                    stats["global"] += 1

            stats["ai_time"] += (time.perf_counter() - t0)
            stats["ai_frames"] += 1

            with frame_lock:
                shared["predictions"][f_id] = pred_point
                shared["processed_ids"].add(f_id)

        # Clean up CUDA and TRT contexts
        del trt_128
        del trt_512
        cuda_ctx.pop()
        del cuda_ctx
        gc.collect()
        shared["ai_stats"] = stats

    # Start AI Worker Thread
    t_ai = threading.Thread(target=ai_worker)
    t_ai.start()

    while not shared["ai_ready"] and shared["is_running"]:
        time.sleep(0.05)

    # --- Main Thread: Strict 30 FPS Live Producer ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)

    cam_fps = 30.0
    frame_interval = 1.0 / cam_fps
    cam_frame_id = 0
    cam_start_time = time.perf_counter()

    try:
        while True:
            t_loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            with frame_lock:
                shared["frame_data"] = frame
                shared["frame_id"] = cam_frame_id

            cam_frame_id += 1

            # Enforce 30 FPS timing
            elapsed = time.perf_counter() - t_loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        time.sleep(0.5)
        shared["is_running"] = False
        t_ai.join()
        cap.release()

    total_cam_time = time.perf_counter() - cam_start_time
    total_cam_frames = cam_frame_id
    ai_stats = shared.get("ai_stats", {"local": 0, "global": 0, "ai_time": 0.0, "ai_frames": 0})
    processed_count = len(shared["processed_ids"])
    dropped_count = total_cam_frames - processed_count
    drop_rate = (dropped_count / total_cam_frames * 100.0) if total_cam_frames > 0 else 0.0

    # Real-Time Accuracy Accounting (Evaluates ALL camera frames)
    TP, FP, FN = 0, 0, 0
    dist_sum, dist_count = 0.0, 0

    for fid in range(total_cam_frames):
        if fid not in GT_DICT:
            continue
        gt_vis = GT_DICT[fid]['vis']
        gt_x = GT_DICT[fid]['x']
        gt_y = GT_DICT[fid]['y']

        # Did the AI process this frame in real time?
        pred = shared["predictions"].get(fid, None)

        if pred is not None:
            pred_x_orig = pred[0] * scale_w
            pred_y_orig = pred[1] * scale_h
            if gt_vis == 1:
                dist = math.hypot(pred_x_orig - gt_x, pred_y_orig - gt_y)
                dist_sum += dist
                dist_count += 1
                if dist <= DIST_THRESH:
                    TP += 1
                else:
                    FP += 1
                    FN += 1
            else:
                FP += 1
        else:
            # FRAME DROPPED BY HARDWARE or UNRECOVERED!
            if gt_vis == 1:
                FN += 1

    precision = (TP / (TP + FP) * 100.0) if (TP + FP) > 0 else 0.0
    recall = (TP / (TP + FN) * 100.0) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_dist = (dist_sum / dist_count) if dist_count > 0 else 0.0

    effective_fps = processed_count / total_cam_time if total_cam_time > 0 else 0.0
    ai_engine_fps = ai_stats["ai_frames"] / ai_stats["ai_time"] if ai_stats["ai_time"] > 0 else 0.0

    tot_passes = ai_stats["local"] + ai_stats["global"]
    loc_ratio = (ai_stats["local"] / tot_passes * 100.0) if tot_passes > 0 else 0.0
    glob_ratio = (ai_stats["global"] / tot_passes * 100.0) if tot_passes > 0 else 0.0

    print("\n--------------------------------------------------------------------------------")
    print("[REAL-TIME REPORT] Variant: {}".format(var_name))
    print("總攝影機幀數 (Camera Ingestion): {} frames @ {:.2f} FPS".format(total_cam_frames, total_cam_frames / total_cam_time))
    print("實質推論幀數 (Processed Frames): {} frames ({:.1f} FPS)".format(processed_count, effective_fps))
    print("硬體丟幀數 (Dropped Frames): {} frames ({:.1f}% 丟幀率)".format(dropped_count, drop_rate))
    print("引擎純推論 FPS (TRT Engine FPS): {:.2f} FPS".format(ai_engine_fps))
    print("局部 128 佔比: {:.1f}% | 全域 512 佔比: {:.1f}%".format(loc_ratio, glob_ratio))
    print("即時精確度 (Precision): {:.2f}%".format(precision))
    print("即時召回率 (Recall): {:.2f}%".format(recall))
    print("即時 F1-Score: {:.2f}%".format(f1))
    print("平均定位誤差 (Mean Error): {:.2f} px".format(mean_dist))
    print("--------------------------------------------------------------------------------\n")

    return {
        "Variant": var_name,
        "Total_Cam_Frames": total_cam_frames,
        "Processed_Frames": processed_count,
        "Dropped_Frames": dropped_count,
        "Drop_Rate(%)": round(drop_rate, 2),
        "Effective_FPS": round(effective_fps, 2),
        "Engine_FPS": round(ai_engine_fps, 2),
        "Local_Ratio(%)": round(loc_ratio, 2),
        "Global_Ratio(%)": round(glob_ratio, 2),
        "Precision(%)": round(precision, 2),
        "Recall(%)": round(recall, 2),
        "F1(%)": round(f1, 2),
        "Mean_Error(px)": round(mean_dist, 2),
        "TP": TP, "FP": FP, "FN": FN
    }

def append_to_csv(row_dict, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def main():
    args = parse_args()

    # If variant is 'all', orchestrate each variant via isolated subprocesses
    if args.variant == "all":
        if os.path.exists(args.output_csv):
            try:
                os.remove(args.output_csv)
            except Exception:
                pass

        variant_order = ["native_global", "baseline", "no_mask", "no_kf", "proposed"]
        for v in variant_order:
            print("\n" + "#"*80)
            print("[ORCHESTRATOR] Launching variant in separate OS process: {}".format(v))
            print("#"*80)
            cmd = [
                sys.executable, __file__,
                "--variant", v,
                "--video", args.video,
                "--csv", args.csv,
                "--output_csv", args.output_csv,
                "--engine_128", args.engine_128,
                "--engine_512", args.engine_512,
                "--motion", str(args.motion),
                "--local_conf", str(args.local_conf),
                "--global_conf", str(args.global_conf),
                "--miss", str(args.miss)
            ]
            ret = subprocess.call(cmd)
            if ret != 0:
                print("[WARNING] Variant {} finished with exit code {}".format(v, ret))
            time.sleep(2)  # Allow OS to reclaim unified memory

        print("\n================================================================================")
        print("JETSON NANO REAL-TIME BENCHMARK COMPLETE!")
        print("Summary written to: {}".format(args.output_csv))
        print("================================================================================")
        return

    # Single variant execution
    v_info = VARIANTS_MAP[args.variant]

    # Load GT
    GT_DICT = {}
    if not os.path.exists(args.csv):
        print("[ERROR] CSV not found: {}".format(args.csv))
        return
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            fkey = 'Frame' if 'Frame' in r else 'frame_num'
            vkey = 'Visibility' if 'Visibility' in r else 'visible'
            xkey = 'X' if 'X' in r else 'x'
            ykey = 'Y' if 'Y' in r else 'y'
            GT_DICT[int(r[fkey])] = {'vis': int(r[vkey]), 'x': float(r[xkey]), 'y': float(r[ykey])}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()

    res = run_realtime_trial(
        v_info["name"],
        v_info["use_kf"],
        v_info["use_mask"],
        v_info["pure_global"],
        args,
        GT_DICT,
        W_orig,
        H_orig
    )

    append_to_csv(res, args.output_csv)

if __name__ == "__main__":
    main()
