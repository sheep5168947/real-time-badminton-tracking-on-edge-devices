# -*- coding: utf-8 -*-
"""
Jetson Nano Benchmark: Asymmetric Hierarchical Coarse-to-Fine Search
Evaluates 128x128 Local Fine Search + 256x144 Fast Global Rescue vs 512x288 Legacy Rescue
under Strict 30 FPS Camera Ingestion.
"""

import os
import sys

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
from collections import deque
import pycuda.driver as cuda
import tensorrt as trt

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

def fast_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_shuttle_center_local(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0:
        return 0, 0, 0.0
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
        padded = np.zeros((size, size, img.shape[2] if len(img.shape) == 3 else 1), dtype=img.dtype)
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        padded[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
        return padded, x_min, y_min
    return cropped, x_min, y_min

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_id = int(row.get("Frame", row.get("frame_num")))
            vis = int(row.get("Visibility", row.get("visible")))
            x = float(row.get("X", row.get("x")))
            y = float(row.get("Y", row.get("y")))
            gt[f_id] = {"vis": vis, "x": x, "y": y}
    return gt

def profile_engine(name, engine_path, input_shape, runs=100):
    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        model = TRTModel(engine_path, ctx)
        dummy = np.random.randn(*input_shape).astype(np.float32)
        # Warmup
        for _ in range(10):
            _ = model(dummy)
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        mean_lat = np.mean(latencies)
        fps = 1000.0 / mean_lat
        print(f"[{name}] Latency: {mean_lat:.2f} ms | FPS: {fps:.2f} FPS")
        return mean_lat, fps
    finally:
        ctx.pop()

def run_streaming_benchmark(video_path, csv_path, engine_128, engine_global, global_res=(256, 144), threshold=0.55, thresh_global=0.55, tag="", save_json=None):
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {tag} | Global Res: {global_res} | Thresh Local: {threshold} | Thresh Global: {thresh_global}")
    print("=" * 80)

    gt_dict = load_ground_truth(csv_path)
    cap = cv2.VideoCapture(video_path)
    w_orig = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    h_orig = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    dist_thresh = 15.0 * (float(w_orig) / 1280.0)

    shared = {
        "frame_id": -1, "frame_data": None, "is_running": True,
        "predictions": {}, "processed_ids": set(),
        "local_cnt": 0, "global_cnt": 0,
        "records": {}
    }
    frame_lock = threading.Lock()

    def ai_worker():
        try:
            cuda.init()
            cuda_ctx = cuda.Device(0).make_context()
            trt_128 = TRTModel(engine_128, cuda_ctx)
            trt_glob = TRTModel(engine_global, cuda_ctx)

            # Warmup
            print(f"[{tag}] Warming up engines...", flush=True)
            _ = trt_128(np.zeros((1, 9, 128, 128), dtype=np.float32))
            _ = trt_glob(np.zeros((1, 9, global_res[1], global_res[0]), dtype=np.float32))
            print(f"[{tag}] Warmup complete, entering inference loop...", flush=True)

            last_id = -1
            ai_queue_512 = deque(maxlen=3)
            ai_queue_glob = deque(maxlen=3)
            is_lost = True
            miss_count = 0
            last_x, last_y = 256, 144

            while shared["is_running"]:
                with frame_lock:
                    f_id = shared["frame_id"]
                    f_data = shared["frame_data"]
                if f_id <= last_id or f_data is None:
                    time.sleep(0.001)
                    continue
                last_id = f_id

                f_512 = cv2.resize(f_data, (512, 288))
                f_glob = cv2.resize(f_data, global_res)
                ai_queue_512.append(f_512)
                ai_queue_glob.append(f_glob)

                if len(ai_queue_512) < 3:
                    continue

                shared["processed_ids"].add(f_id)
                pred_pt = None

                if not is_lost:
                    # LOCAL 128 SEARCH (on 512x288 coordinate space)
                    shared["local_cnt"] += 1
                    px, py = int(last_x), int(last_y)

                    c0, _, _ = crop_and_pad(ai_queue_512[0], px, py, 128)
                    c1, _, _ = crop_and_pad(ai_queue_512[1], px, py, 128)
                    c2, _, _ = crop_and_pad(ai_queue_512[2], px, py, 128)

                    imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
                    tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                    raw_out = trt_128(tx)
                    hm = fast_sigmoid(raw_out).squeeze()

                    lx, ly, conf = get_shuttle_center_local(hm)
                    if conf > threshold:
                        is_lost = False
                        miss_count = 0
                        dx, dy = px - 64 + lx, py - 64 + ly
                        pred_pt = (dx, dy)
                        last_x, last_y = dx, dy
                    else:
                        miss_count += 1
                        if miss_count >= 3:
                            is_lost = True
                else:
                    # FAST GLOBAL RESCUE SEARCH
                    shared["global_cnt"] += 1
                    imgs_g = np.concatenate((ai_queue_glob[0], ai_queue_glob[1], ai_queue_glob[2]), axis=-1).astype(np.float32) / 255.0
                    tx_g = np.ascontiguousarray(np.transpose(imgs_g, (2, 0, 1))[np.newaxis, ...])
                    raw_out = trt_glob(tx_g)
                    hm = fast_sigmoid(raw_out).squeeze()

                    gx, gy, conf = get_shuttle_center_local(hm)
                    if conf > thresh_global:
                        is_lost = False
                        miss_count = 0
                        # Map back to 512x288 coordinate space
                        factor_x = 512.0 / float(global_res[0])
                        factor_y = 288.0 / float(global_res[1])
                        gx_512 = gx * factor_x
                        gy_288 = gy * factor_y
                        pred_pt = (gx_512, gy_288)
                        last_x, last_y = gx_512, gy_288

                shared["predictions"][f_id] = pred_pt
                shared["records"][f_id] = {
                    "mode": "LOCAL" if (pred_pt is not None and not is_lost) else "GLOBAL",
                    "pred": [float(pred_pt[0]), float(pred_pt[1])] if pred_pt is not None else None,
                    "crop": [int(last_x), int(last_y)]
                }

            cuda_ctx.pop()
        except Exception as e:
            import traceback
            print(f"[{tag}] AI WORKER CRASHED WITH EXCEPTION:", flush=True)
            traceback.print_exc()

    t_ai = threading.Thread(target=ai_worker)
    t_ai.daemon = True
    t_ai.start()

    # Hardware camera producer at strict 30 FPS
    frame_interval = 1.0 / 30.0
    cam_id = 0
    t_cam_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            shared["frame_data"] = frame
            shared["frame_id"] = cam_id
        cam_id += 1
        elapsed = time.perf_counter() - t0
        st = frame_interval - elapsed
        if st > 0:
            time.sleep(st)

    time.sleep(0.5)
    shared["is_running"] = False
    t_ai.join()
    t_total_cam = time.perf_counter() - t_cam_start
    cap.release()

    # Hardware accounting & metrics calculation
    total_cam_frames = cam_id
    processed_count = len(shared["processed_ids"])
    dropped_count = total_cam_frames - processed_count
    drop_rate = (dropped_count / total_cam_frames * 100.0) if total_cam_frames > 0 else 0.0
    effective_fps = processed_count / t_total_cam if t_total_cam > 0 else 0.0

    loc_cnt = shared["local_cnt"]
    glob_cnt = shared["global_cnt"]
    total_inferences = loc_cnt + glob_cnt
    loc_ratio = (loc_cnt / total_inferences * 100.0) if total_inferences > 0 else 0.0
    glob_ratio = (glob_cnt / total_inferences * 100.0) if total_inferences > 0 else 0.0

    TP, FP, FN = 0, 0, 0
    errors = []
    for fid in range(total_cam_frames):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gt_x = gt_dict[fid]["x"]
        gt_y = gt_dict[fid]["y"]

        pred = shared["predictions"].get(fid, None)
        if pred is not None:
            px_orig = pred[0] * scale_w
            py_orig = pred[1] * scale_h
            if gvis == 1:
                dist = math.hypot(px_orig - gt_x, py_orig - gt_y)
                if dist <= dist_thresh:
                    TP += 1
                    errors.append(dist)
                else:
                    FP += 1
            else:
                FP += 1
        else:
            if gvis == 1:
                FN += 1

    prec = (TP / (TP + FP) * 100.0) if (TP + FP) > 0 else 0.0
    rec = (TP / (TP + FN) * 100.0) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mean_err = np.mean(errors) if errors else 0.0

    print(f"Results for [{tag}]:")
    print(f"  Camera Ingested: {total_cam_frames} frames ({t_total_cam:.2f}s)")
    print(f"  Processed Frames: {processed_count} | Dropped: {dropped_count} ({drop_rate:.2f}%)")
    print(f"  Effective Camera FPS: {effective_fps:.2f} FPS")
    print(f"  Local Search Ratio: {loc_ratio:.2f}% | Global Rescue Ratio: {glob_ratio:.2f}%")
    print(f"  Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1-Score: {f1:.2f}%")
    print(f"  Mean Error: {mean_err:.2f} px (TP={TP}, FP={FP}, FN={FN})")

    if save_json is not None:
        import json
        save_data = {
            "tag": tag,
            "total_cam_frames": total_cam_frames,
            "processed_count": processed_count,
            "dropped_count": dropped_count,
            "drop_rate": drop_rate,
            "effective_fps": effective_fps,
            "processed_ids": list(shared["processed_ids"]),
            "records": shared["records"]
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(save_data, f)
        print(f"  [SAVED] Benchmark records written to {save_json}")

    return {
        "tag": tag,
        "global_res": f"{global_res[0]}x{global_res[1]}",
        "cam_frames": total_cam_frames,
        "processed": processed_count,
        "dropped": dropped_count,
        "drop_rate": drop_rate,
        "effective_fps": effective_fps,
        "local_ratio": loc_ratio,
        "global_ratio": glob_ratio,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_error": mean_err,
        "TP": TP, "FP": FP, "FN": FN
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="256", choices=["all", "256", "512", "profile"])
    parser.add_argument("--video", type=str, default="weights/1_01_00.mp4")
    parser.add_argument("--csv", type=str, default="weights/gt.csv")
    parser.add_argument("--engine_128", type=str, default="weights/miniroi_128x128.engine")
    parser.add_argument("--engine_256", type=str, default="weights/miniroi_256x144.engine")
    parser.add_argument("--engine_512", type=str, default="weights/miniroi_512x288.engine")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--thresh_global", type=float, default=0.35)
    args = parser.parse_args()

    if args.mode in ["all", "profile"]:
        print("\n" + "#" * 80)
        print("STEP 1: ENGINE LATENCY PROFILING ON JETSON NANO")
        print("#" * 80)
        profile_engine("128x128 Local Engine", args.engine_128, (1, 9, 128, 128))
        profile_engine("256x144 Global Engine", args.engine_256, (1, 9, 144, 256))
        profile_engine("512x288 Global Engine", args.engine_512, (1, 9, 288, 512))

    res_512, res_256 = None, None

    if args.mode in ["all", "512"]:
        res_512 = run_streaming_benchmark(
            args.video, args.csv, args.engine_128, args.engine_512,
            global_res=(512, 288), threshold=args.threshold,
            tag="Legacy MiniROI (128 Local + 512 Global Rescue)",
            save_json="records_512.json"
        )

    if args.mode in ["all", "256"]:
        res_256 = run_streaming_benchmark(
            args.video, args.csv, args.engine_128, args.engine_256,
            global_res=(256, 144), threshold=args.threshold, thresh_global=args.thresh_global,
            tag="Proposed Hierarchical (128 Local + 256 Fast Rescue)",
            save_json="records_256.json"
        )

    if res_512 and res_256:
        print("\n" + "#" * 80)
        print("FINAL HEAD-TO-HEAD SUMMARY (JETSON NANO STRICT REAL-TIME)")
        print("#" * 80)
        print(f"{'Metric':<25} | {'Legacy (512 Rescue)':<22} | {'Proposed (256 Rescue)':<22} | {'Improvement'}")
        print("-" * 85)
        print(f"{'Dropped Frames':<25} | {res_512['dropped']:<22} | {res_256['dropped']:<22} | {res_512['dropped'] - res_256['dropped']} fewer frames")
        print(f"{'Drop Rate (%)':<25} | {res_512['drop_rate']:<21.2f}% | {res_256['drop_rate']:<21.2f}% | {res_512['drop_rate'] - res_256['drop_rate']:+.2f}%")
        print(f"{'Effective Camera FPS':<25} | {res_512['effective_fps']:<22.2f} | {res_256['effective_fps']:<22.2f} | {res_256['effective_fps'] - res_512['effective_fps']:+.2f} FPS")
        print(f"{'Precision (%)':<25} | {res_512['precision']:<21.2f}% | {res_256['precision']:<21.2f}% | {res_256['precision'] - res_512['precision']:+.2f}%")
        print(f"{'Recall (%)':<25} | {res_512['recall']:<21.2f}% | {res_256['recall']:<21.2f}% | {res_256['recall'] - res_512['recall']:+.2f}%")
        print(f"{'F1-Score (%)':<25} | {res_512['f1']:<21.2f}% | {res_256['f1']:<21.2f}% | {res_256['f1'] - res_512['f1']:+.2f}%")
        print(f"{'Mean Error (px)':<25} | {res_512['mean_error']:<22.2f} | {res_256['mean_error']:<22.2f} | {res_256['mean_error'] - res_512['mean_error']:+.2f} px")
        print("#" * 80)

if __name__ == "__main__":
    main()
