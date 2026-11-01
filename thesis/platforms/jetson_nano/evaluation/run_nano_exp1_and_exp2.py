# -*- coding: utf-8 -*-
"""
Jetson Nano Real-Time Experiment 1 & Experiment 2 Hardware Benchmark
Evaluates under STRICT 30 FPS camera ingestion with TensorRT execution.

Experiment 1:
- Local 128x128 search edge-case analysis when ball leaves crop (出框)
- Monitors false positive occurrences and whether Motion Mask blocks them.

Experiment 2:
- Confidence threshold ablation (Threshold=0.55 vs Threshold=0.00 / Argmax)
- Cross-evaluated with Motion Mask (ON vs OFF)
- Measures Real-Time FPS, Hardware Frame Drop Count & Rate (%), Precision, Recall, F1
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

def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano Real-Time Hardware Benchmark")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "exp1", "exp2", "trial_single"],
                        help="Benchmark mode")
    parser.add_argument("--video", type=str, default="weights/1_01_00.mp4", help="Video file path")
    parser.add_argument("--csv", type=str, default="weights/gt.csv", help="Ground Truth CSV file path")
    parser.add_argument("--engine_128", type=str, default="weights/miniroi_128x128.engine")
    parser.add_argument("--engine_512", type=str, default="weights/miniroi_512x288.engine")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--use_mask", action="store_true")
    parser.add_argument("--tag", type=str, default="trial")
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
    if max_val == 0: return 0, 0, 0.0
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

# =============================================================================
# EXPERIMENT 1: LOCAL SEARCH REAL-TIME EDGE-CASE ANALYSIS ON NANO
# =============================================================================
def run_nano_exp1(args):
    print("\n" + "=" * 80)
    print("[NANO REAL-TIME EXPERIMENT 1: LOCAL 128x128 EDGE-CASE & MASK ANALYSIS]")
    print("=" * 80)

    gt_dict = load_ground_truth(args.csv)
    cap = cv2.VideoCapture(args.video)
    w_orig = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    h_orig = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0

    local_crops_count = 0
    ball_out_of_crop_count = 0
    raw_noise_ge_55_count = 0
    mask_blocked_case2_count = 0
    mask_rescued_case1_count = 0
    mask_killed_ball_count = 0

    shared = {
        "frame_id": -1, "frame_data": None, "is_running": True,
        "processed": set()
    }
    frame_lock = threading.Lock()

    def ai_worker():
        nonlocal local_crops_count, ball_out_of_crop_count, raw_noise_ge_55_count
        nonlocal mask_blocked_case2_count, mask_rescued_case1_count, mask_killed_ball_count

        cuda.init()
        cuda_ctx = cuda.Device(0).make_context()
        trt_128 = TRTModel(args.engine_128, cuda_ctx)
        trt_512 = TRTModel(args.engine_512, cuda_ctx)

        # Warmup inside worker thread
        _ = trt_128(np.zeros((1, 9, 128, 128), dtype=np.float32))
        _ = trt_512(np.zeros((1, 9, 288, 512), dtype=np.float32))

        last_id = -1
        ai_queue = deque(maxlen=3)
        is_lost = True
        miss_count = 0
        kf = None
        last_x, last_y = 256, 144

        while shared["is_running"]:
            with frame_lock:
                f_id = shared["frame_id"]
                f_data = shared["frame_data"]
            if f_id <= last_id or f_data is None:
                time.sleep(0.001)
                continue
            last_id = f_id

            f_train = cv2.resize(f_data, (512, 288))
            ai_queue.append(f_train)
            if len(ai_queue) < 3:
                continue

            shared["processed"].add(f_id)
            img0, img1, img2 = ai_queue[0], ai_queue[1], ai_queue[2]
            gt = gt_dict.get(f_id, {"vis": 0, "x": 0, "y": 0})
            gx = gt["x"] / scale_w if gt["vis"] == 1 else -1000
            gy = gt["y"] / scale_h if gt["vis"] == 1 else -1000

            if not is_lost:
                # LOCAL SEARCH PHASE
                local_crops_count += 1
                if kf is not None:
                    p = kf.predict()
                    px, py = int(p[0, 0]), int(p[1, 0])
                else:
                    px, py = int(last_x), int(last_y)

                c0, _, _ = crop_and_pad(img0, px, py, 128)
                c1, _, _ = crop_and_pad(img1, px, py, 128)
                c2, _, _ = crop_and_pad(img2, px, py, 128)

                gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                _, mc = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                mf = mc.astype(np.float32) / 255.0

                imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
                tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                raw_out = trt_128(tx)
                raw_hm = fast_sigmoid(raw_out).squeeze()
                masked_hm = raw_hm * mf

                rx, ry, rc = get_shuttle_center_local(raw_hm)
                mx, my, mc = get_shuttle_center_local(masked_hm)

                ball_in_crop = (gt["vis"] == 1 and abs(gx - px) < 60 and abs(gy - py) < 60)

                if not ball_in_crop:
                    ball_out_of_crop_count += 1
                    if rc >= 0.55:
                        raw_noise_ge_55_count += 1
                        if mc < 0.55:
                            mask_blocked_case2_count += 1
                            print(f"  [CASE 2 BLOCKED ON NANO!] Frame {f_id}: raw={rc:.3f} -> masked={mc:.3f}", flush=True)
                        else:
                            print(f"  [RAW NOISE >= 0.55 NOT BLOCKED!] Frame {f_id}: raw={rc:.3f}, masked={mc:.3f} (moving noise!)", flush=True)
                else:
                    # Ball is in crop
                    local_gx = 64 + (gx - px)
                    local_gy = 64 + (gy - py)
                    err_r = np.hypot(rx - local_gx, ry - local_gy)
                    err_m = np.hypot(mx - local_gx, my - local_gy)

                    # Check Case 1
                    if err_r > 15.0 and err_m <= 6.0 and mc >= 0.35:
                        mask_rescued_case1_count += 1
                        print(f"  [CASE 1 RESCUED ON NANO!] Frame {f_id}: err_r={err_r:.1f} -> err_m={err_m:.1f}", flush=True)

                    # Check if Mask killed correct detection
                    if err_r <= 6.0 and rc >= 0.55 and (mc < 0.55 or err_m > 10.0):
                        mask_killed_ball_count += 1
                        print(f"  [MASK KILLED BALL ON NANO!] Frame {f_id}: raw_conf={rc:.3f} -> masked={mc:.3f}", flush=True)

                if rc > 0.55:
                    is_lost = False
                    miss_count = 0
                    dx, dy = px - 64 + rx, py - 64 + ry
                    last_x, last_y = dx, dy
                    if kf is None: kf = init_kalman_filter(dx, dy)
                    else: kf.correct(np.array([[dx], [dy]], dtype=np.float32))
                else:
                    miss_count += 1
                    if miss_count >= 3:
                        is_lost = True
                        kf = None
            else:
                # GLOBAL SEARCH PHASE
                imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
                tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                raw_out = trt_512(tx)
                raw_hm = fast_sigmoid(raw_out).squeeze()
                gx_det, gy_det, gconf = get_shuttle_center_local(raw_hm)
                if gconf > 0.55:
                    is_lost = False
                    miss_count = 0
                    last_x, last_y = gx_det, gy_det
                    kf = init_kalman_filter(gx_det, gy_det)
        cuda_ctx.pop()

    t_ai = threading.Thread(target=ai_worker)
    t_ai.daemon = True
    t_ai.start()

    # Ingestion thread at strict 30 FPS
    frame_interval = 1.0 / 30.0
    cam_id = 0
    t_cam_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        with frame_lock:
            shared["frame_data"] = frame
            shared["frame_id"] = cam_id
        cam_id += 1
        elapsed = time.perf_counter() - t0
        st = frame_interval - elapsed
        if st > 0: time.sleep(st)

    time.sleep(0.5)
    shared["is_running"] = False
    t_ai.join()
    cap.release()

    print("\n" + "=" * 80)
    print("NANO REAL-TIME EXPERIMENT 1 FINAL RESULTS:")
    print("=" * 80)
    print(f"Total Local 128x128 Crops Ingested: {local_crops_count}")
    print(f"Ball OUT of 128x128 Crop Count: {ball_out_of_crop_count}")
    print(f"Frames where Raw Noise >= 0.55 in empty crop: {raw_noise_ge_55_count}")
    print(f"Case 2 (False Alarm Blocked by Mask): {mask_blocked_case2_count}")
    print(f"Case 1 (Peak Rescued by Mask): {mask_rescued_case1_count}")
    print(f"Mask Hurt / Killed Correct Ball Detections: {mask_killed_ball_count}")
    print("=" * 80)

# =============================================================================
# EXPERIMENT 2: THRESHOLD & MASK REAL-TIME ABLATION ON NANO
# =============================================================================
def run_single_realtime_trial(args):
    print("\n" + "-" * 70)
    print(f"[TRIAL] Tag: {args.tag} | Threshold: {args.threshold} | Mask: {args.use_mask}")
    print("-" * 70)

    gt_dict = load_ground_truth(args.csv)
    cap = cv2.VideoCapture(args.video)
    w_orig = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
    h_orig = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    dist_thresh = 15.0 * (float(w_orig) / 1280.0)

    shared = {
        "frame_id": -1, "frame_data": None, "is_running": True,
        "predictions": {}, "processed_ids": set(),
        "local_cnt": 0, "global_cnt": 0
    }
    frame_lock = threading.Lock()

    def ai_worker():
        cuda.init()
        cuda_ctx = cuda.Device(0).make_context()
        trt_128 = TRTModel(args.engine_128, cuda_ctx)
        trt_512 = TRTModel(args.engine_512, cuda_ctx)

        # Warmup inside worker thread
        _ = trt_128(np.zeros((1, 9, 128, 128), dtype=np.float32))
        _ = trt_512(np.zeros((1, 9, 288, 512), dtype=np.float32))

        last_id = -1
        ai_queue = deque(maxlen=3)
        is_lost = True
        miss_count = 0
        kf = None
        last_x, last_y = 256, 144

        while shared["is_running"]:
            with frame_lock:
                f_id = shared["frame_id"]
                f_data = shared["frame_data"]
            if f_id <= last_id or f_data is None:
                time.sleep(0.001)
                continue
            last_id = f_id

            f_train = cv2.resize(f_data, (512, 288))
            ai_queue.append(f_train)
            if len(ai_queue) < 3:
                continue

            shared["processed_ids"].add(f_id)
            img0, img1, img2 = ai_queue[0], ai_queue[1], ai_queue[2]
            pred_pt = None

            if not is_lost:
                # LOCAL 128
                shared["local_cnt"] += 1
                if kf is not None:
                    p = kf.predict()
                    px, py = int(p[0, 0]), int(p[1, 0])
                else:
                    px, py = int(last_x), int(last_y)

                c0, _, _ = crop_and_pad(img0, px, py, 128)
                c1, _, _ = crop_and_pad(img1, px, py, 128)
                c2, _, _ = crop_and_pad(img2, px, py, 128)

                imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
                tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                raw_out = trt_128(tx)
                hm = fast_sigmoid(raw_out).squeeze()

                if args.use_mask:
                    gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                    gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                    _, mc = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                    hm = hm * (mc.astype(np.float32) / 255.0)

                lx, ly, conf = get_shuttle_center_local(hm)
                has_det = (conf > args.threshold) if args.threshold > 0 else (conf > 0.0001)

                if has_det:
                    is_lost = False
                    miss_count = 0
                    dx, dy = px - 64 + lx, py - 64 + ly
                    pred_pt = (dx, dy)
                    last_x, last_y = dx, dy
                    if kf is None: kf = init_kalman_filter(dx, dy)
                    else: kf.correct(np.array([[dx], [dy]], dtype=np.float32))
                else:
                    miss_count += 1
                    if miss_count >= 3:
                        is_lost = True
                        kf = None
            else:
                # GLOBAL 512
                shared["global_cnt"] += 1
                imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
                tx = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
                raw_out = trt_512(tx)
                hm = fast_sigmoid(raw_out).squeeze()

                if args.use_mask:
                    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    _, mg = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                    hm = hm * cv2.resize(mg.astype(np.float32) / 255.0, (512, 288))

                gx_det, gy_det, conf = get_shuttle_center_local(hm)
                has_det = (conf > args.threshold) if args.threshold > 0 else (conf > 0.0001)

                if has_det:
                    is_lost = False
                    miss_count = 0
                    pred_pt = (gx_det, gy_det)
                    last_x, last_y = gx_det, gy_det
                    kf = init_kalman_filter(gx_det, gy_det)

            shared["predictions"][f_id] = pred_pt
        cuda_ctx.pop()

    t_ai = threading.Thread(target=ai_worker)
    t_ai.daemon = True
    t_ai.start()

    frame_interval = 1.0 / 30.0
    cam_id = 0
    t_cam_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        with frame_lock:
            shared["frame_data"] = frame
            shared["frame_id"] = cam_id
        cam_id += 1
        elapsed = time.perf_counter() - t0
        st = frame_interval - elapsed
        if st > 0: time.sleep(st)

    time.sleep(0.5)
    shared["is_running"] = False
    t_ai.join()
    t_total_cam = time.perf_counter() - t_cam_start
    cap.release()

    # Hardware Accounting
    total_cam_frames = cam_id
    processed_count = len(shared["processed_ids"])
    dropped_count = total_cam_frames - processed_count
    drop_rate = (dropped_count / total_cam_frames * 100.0) if total_cam_frames > 0 else 0.0
    effective_fps = processed_count / t_total_cam if t_total_cam > 0 else 0.0

    TP, FP, FN = 0, 0, 0
    for fid in range(total_cam_frames):
        if fid not in gt_dict: continue
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

    total_ai_runs = shared["local_cnt"] + shared["global_cnt"]
    local_ratio = (shared["local_cnt"] / total_ai_runs * 100.0) if total_ai_runs > 0 else 0.0

    print(f"[RESULT {args.tag}] Frames={total_cam_frames}, Processed={processed_count}, Dropped={dropped_count} ({drop_rate:.1f}%) | Eff.FPS={effective_fps:.1f}")
    print(f"[ACCURACY {args.tag}] TP={TP}, FP={FP}, FN={FN} | Prec={prec:.2f}%, Rec={rec:.2f}%, F1={f1:.2f}% | Local128={local_ratio:.1f}%")

def main():
    args = parse_args()

    if args.mode == "exp1":
        run_nano_exp1(args)
    elif args.mode == "trial_single":
        run_single_realtime_trial(args)
    elif args.mode == "all":
        # Run Exp 1
        print("\n" + "=" * 80)
        print("STARTING SUITE: EXP 1 (LOCAL EDGE-CASE) & EXP 2 (THRESHOLD ABLATION) ON NANO")
        print("=" * 80)
        cmd_exp1 = [sys.executable, __file__, "--mode", "exp1",
                    "--video", args.video, "--csv", args.csv,
                    "--engine_128", args.engine_128, "--engine_512", args.engine_512]
        subprocess.run(cmd_exp1, check=True)

        # Run Exp 2 (4 Subprocesses to prevent TRT OOM)
        exp2_configs = [
            ("T=0.55_Mask=OFF", 0.55, False),
            ("T=0.55_Mask=ON ", 0.55, True),
            ("T=0.00_Mask=OFF", 0.00, False),
            ("T=0.00_Mask=ON ", 0.00, True),
        ]
        for tag, th, um in exp2_configs:
            cmd = [sys.executable, __file__, "--mode", "trial_single",
                   "--video", args.video, "--csv", args.csv,
                   "--engine_128", args.engine_128, "--engine_512", args.engine_512,
                   "--threshold", str(th), "--tag", tag]
            if um: cmd.append("--use_mask")
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
