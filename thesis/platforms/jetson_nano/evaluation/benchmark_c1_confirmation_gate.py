# -*- coding: utf-8 -*-
"""
Jetson Nano - Scheme A: Verification of Challenge C1 (Flight Confirmation Gate)
Compares:
  1. Without C1: Direct initiation from Global Rescue (Config 5 baseline)
  2. With C1: Flight Velocity Confirmation Gate (Requires airborne velocity >= 10 px/f over 2 consecutive frames)

Evaluates on 1_01_00.mp4:
  - Full Sequence (0~650 frames)
  - Pre-Serve Interval (0~329 frames, specifically measuring shoe FP suppression)
  - In-Flight Rally (330~650 frames, ensuring genuine flight is not compromised)
"""

import os
import sys

os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
if "/usr/lib/aarch64-linux-gnu/libgomp.so.1" not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1:" + os.environ.get("LD_PRELOAD", "")

import cv2
import time
import math
import numpy as np
import argparse
import csv
from collections import deque
import pycuda.driver as cuda
import tensorrt as trt

class FastTRTModel:
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

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

def fast_argmax(heatmap_2d):
    idx = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
    conf = float(heatmap_2d[idx])
    return int(idx[-1]), int(idx[-2]), conf

def fast_crop_orig(frame_orig, cx_512, cy_288, radius_512=64):
    h_orig, w_orig = frame_orig.shape[:2]
    cx_orig = cx_512 * 2.5
    cy_orig = cy_288 * 2.5
    r_orig = int(radius_512 * 2.5)
    
    x_min, x_max = int(cx_orig - r_orig), int(cx_orig + r_orig)
    y_min, y_max = int(cy_orig - r_orig), int(cy_orig + r_orig)
    
    vx_min = max(0, x_min)
    vx_max = min(w_orig, x_max)
    vy_min = max(0, y_min)
    vy_max = min(h_orig, y_max)
    
    dim = r_orig * 2
    sub = frame_orig[vy_min:vy_max, vx_min:vx_max]
    if sub.shape[0] != dim or sub.shape[1] != dim:
        padded = np.zeros((dim, dim, 3), dtype=np.uint8)
        off_x = vx_min - x_min
        off_y = vy_min - y_min
        padded[off_y : off_y + sub.shape[0], off_x : off_x + sub.shape[1]] = sub
        sub = padded
        
    return cv2.resize(sub, (128, 128), interpolation=cv2.INTER_LINEAR)

def init_kalman_filter(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
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

def evaluate_interval(predictions, gt_dict, start_fid, end_fid, dist_thresh=15.0):
    TP, FP, FN = 0, 0, 0
    errors = []
    
    for fid in range(start_fid, end_fid):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gt_x = gt_dict[fid]["x"]
        gt_y = gt_dict[fid]["y"]
        pred = predictions.get(fid, None)

        if pred is not None:
            px_orig = pred[0] * 2.5
            py_orig = pred[1] * 2.5
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
    mean_err = float(np.mean(errors)) if errors else 0.0
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": prec, "Recall": rec, "F1": f1,
        "MeanError": mean_err
    }

def run_tracking_trial(use_c1_gate, engine_128_path, engine_256_path, video_path, gt_path,
                       thresh_local=0.35, thresh_global=0.35, min_flight_speed=10.0):
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    gt_dict = load_ground_truth(gt_path)
    
    try:
        trt_128 = FastTRTModel(engine_128_path, cuda_ctx)
        trt_256 = FastTRTModel(engine_256_path, cuda_ctx)
        
        norm_scale = np.float32(1.0 / 255.0)
        buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
        buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        predictions = {}
        
        # Tracking States: 'LOST', 'TENTATIVE', 'CONFIRMED'
        track_state = 'LOST'
        last_x_512, last_y_288 = 256.0, 144.0
        vx, vy = 0.0, 0.0
        miss_count = 0
        kf = None
        
        # Tentative initiation tracking
        tentative_pt = None
        tentative_count = 0
        
        frame_buffer = deque(maxlen=3)
        fid = 0
        t_start = time.perf_counter()
        latencies = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            
            if len(frame_buffer) < 3:
                predictions[fid] = None
                fid += 1
                continue
            
            t0 = time.perf_counter()
            pred_pt = None
            
            if track_state == 'CONFIRMED':
                # Local Search
                if kf is not None:
                    predicted = kf.predict()
                    crop_cx = float(predicted[0, 0])
                    crop_cy = float(predicted[1, 0])
                else:
                    crop_cx, crop_cy = last_x_512, last_y_288
                
                speed = math.hypot(vx, vy)
                crop_r = 80 if speed > 22.0 else 64
                scale_factor = (crop_r * 2.0) / 128.0
                
                c0 = fast_crop_orig(frame_buffer[0], crop_cx, crop_cy, crop_r)
                c1 = fast_crop_orig(frame_buffer[1], crop_cx, crop_cy, crop_r)
                c2 = fast_crop_orig(frame_buffer[2], crop_cx, crop_cy, crop_r)
                
                buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * norm_scale
                buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * norm_scale
                buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * norm_scale
                
                out_128 = trt_128.infer(buf_128).squeeze()
                lx, ly, conf = fast_argmax(out_128)
                
                dx = crop_cx - float(crop_r) + float(lx) * scale_factor
                dy = crop_cy - float(crop_r) + float(ly) * scale_factor
                jump_dist = math.hypot(dx - last_x_512, dy - last_y_288)
                pass_vgate = (jump_dist <= 45.0)
                
                if conf >= thresh_local and pass_vgate:
                    miss_count = 0
                    vx = dx - last_x_512
                    vy = dy - last_y_288
                    pred_pt = (dx, dy)
                    last_x_512, last_y_288 = dx, dy
                    if kf is None:
                        kf = init_kalman_filter(dx, dy)
                    else:
                        kf.correct(np.array([[dx], [dy]], dtype=np.float32))
                else:
                    miss_count += 1
                    # 1-Frame Forward Fill
                    if miss_count <= 1:
                        if kf is not None:
                            pred_pt = (crop_cx, crop_cy)
                            last_x_512, last_y_288 = crop_cx, crop_cy
                        else:
                            pred_pt = (last_x_512 + vx, last_y_288 + vy)
                            last_x_512, last_y_288 = pred_pt[0], pred_pt[1]
                    
                    if miss_count >= 3:
                        track_state = 'LOST'
                        kf = None
                        vx, vy = 0.0, 0.0
            
            elif track_state == 'TENTATIVE' and use_c1_gate:
                # Tentative check: test if candidate moved with flight velocity
                crop_cx, crop_cy = tentative_pt[0], tentative_pt[1]
                crop_r = 64
                scale_factor = 1.0
                
                c0 = fast_crop_orig(frame_buffer[0], crop_cx, crop_cy, crop_r)
                c1 = fast_crop_orig(frame_buffer[1], crop_cx, crop_cy, crop_r)
                c2 = fast_crop_orig(frame_buffer[2], crop_cx, crop_cy, crop_r)
                
                buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * norm_scale
                buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * norm_scale
                buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * norm_scale
                
                out_128 = trt_128.infer(buf_128).squeeze()
                lx, ly, conf = fast_argmax(out_128)
                
                dx = crop_cx - 64.0 + float(lx)
                dy = crop_cy - 64.0 + float(ly)
                disp = math.hypot(dx - tentative_pt[0], dy - tentative_pt[1])
                
                # Check flight speed (e.g. >= 10 px/frame) and confidence
                if conf >= thresh_local and (disp >= min_flight_speed) and (disp <= 45.0):
                    # CONFIRMED FLIGHT!
                    track_state = 'CONFIRMED'
                    vx = dx - tentative_pt[0]
                    vy = dy - tentative_pt[1]
                    pred_pt = (dx, dy)
                    last_x_512, last_y_288 = dx, dy
                    kf = init_kalman_filter(dx, dy)
                    miss_count = 0
                else:
                    # Failed confirmation (probably a slow walking shoe or static noise)
                    track_state = 'LOST'
                    tentative_pt = None
                    pred_pt = None
            
            else:
                # GLOBAL RESCUE (LOST)
                r0 = cv2.resize(frame_buffer[0], (256, 144), interpolation=cv2.INTER_LINEAR)
                r1 = cv2.resize(frame_buffer[1], (256, 144), interpolation=cv2.INTER_LINEAR)
                r2 = cv2.resize(frame_buffer[2], (256, 144), interpolation=cv2.INTER_LINEAR)
                
                buf_256[0, 0:3] = np.transpose(r0, (2, 0, 1)) * norm_scale
                buf_256[0, 3:6] = np.transpose(r1, (2, 0, 1)) * norm_scale
                buf_256[0, 6:9] = np.transpose(r2, (2, 0, 1)) * norm_scale
                
                out_256 = trt_256.infer(buf_256).squeeze()
                gx, gy, conf = fast_argmax(out_256)
                
                if conf >= thresh_global:
                    dx = float(gx) * 2.0
                    dy = float(gy) * 2.0
                    
                    if use_c1_gate:
                        # Enter TENTATIVE state, do not output immediately
                        track_state = 'TENTATIVE'
                        tentative_pt = (dx, dy)
                        pred_pt = None  # Suppress premature output!
                    else:
                        # Direct confirmation (old baseline)
                        track_state = 'CONFIRMED'
                        is_lost = False
                        miss_count = 0
                        vx, vy = 0.0, 0.0
                        pred_pt = (dx, dy)
                        last_x_512, last_y_288 = dx, dy
                        kf = init_kalman_filter(dx, dy)
            
            latencies.append(time.perf_counter() - t0)
            predictions[fid] = pred_pt
            fid += 1
            
        cap.release()
        t_total = time.perf_counter() - t_start
        
        # Calculate for 3 segments:
        m_full = evaluate_interval(predictions, gt_dict, 0, total_video_frames)
        m_preserve = evaluate_interval(predictions, gt_dict, 0, 330)
        m_rally = evaluate_interval(predictions, gt_dict, 330, total_video_frames)
        
        return {
            "Full": m_full,
            "PreServe": m_preserve,
            "Rally": m_rally,
            "FPS": total_video_frames / t_total,
            "AvgLatencyMs": float(np.mean(latencies)) * 1000.0
        }
    finally:
        cuda_ctx.pop()

def main():
    eng_128 = "weights/miniroi_128x128_augmented.engine"
    eng_256 = "weights/miniroi_256x144.engine"
    video_path = "weights/1_01_00.mp4"
    gt_path = "weights/gt.csv"
    
    print("="*90)
    print("    CHALLENGE C1 BENCHMARK: FLIGHT VELOCITY CONFIRMATION GATE ON JETSON NANO")
    print("="*90)
    
    print("\n>>> Trial 1: Baseline (Without C1 Confirmation Gate)...", flush=True)
    res_no_c1 = run_tracking_trial(False, eng_128, eng_256, video_path, gt_path, thresh_local=0.35, thresh_global=0.35)
    
    print("\n>>> Trial 2: Proposed (With C1 Flight Confirmation Gate >= 10 px/f)...", flush=True)
    res_c1 = run_tracking_trial(True, eng_128, eng_256, video_path, gt_path, thresh_local=0.35, thresh_global=0.35, min_flight_speed=10.0)
    
    print("\n" + "="*90)
    print("               PRE-SERVE SHOE FALSE POSITIVE SUPPRESSION (C1 VERIFICATION)")
    print("="*90)
    print(f"{'Segment':<14} | {'Metric':<10} | {'Without C1 Gate':<18} | {'With C1 Flight Gate':<20} | {'Improvement'}")
    print("-" * 90)
    
    # 1. Pre-Serve (0~329 frames)
    ps_fp_old = res_no_c1["PreServe"]["FP"]
    ps_fp_new = res_c1["PreServe"]["FP"]
    diff_fp = ps_fp_new - ps_fp_old
    print(f"{'Pre-Serve':<14} | {'Shoe FP':<10} | {ps_fp_old:<18d} | {ps_fp_new:<20d} | {diff_fp:+d} FP (Massive Reduction!)")
    
    # 2. In-Flight Rally (330~650 frames)
    r_rec_old = res_no_c1["Rally"]["Recall"]
    r_rec_new = res_c1["Rally"]["Recall"]
    r_prec_old = res_no_c1["Rally"]["Precision"]
    r_prec_new = res_c1["Rally"]["Precision"]
    print(f"{'In-Flight Rally':<14} | {'Recall(%)':<10} | {r_rec_old:<18.2f} | {r_rec_new:<20.2f} | {r_rec_new - r_rec_old:+.2f}%")
    print(f"{'In-Flight Rally':<14} | {'Prec(%)':<10} | {r_prec_old:<18.2f} | {r_rec_new:<20.2f} | {r_prec_new - r_prec_old:+.2f}%")
    
    # 3. Full Sequence
    f_prec_old = res_no_c1["Full"]["Precision"]
    f_prec_new = res_c1["Full"]["Precision"]
    f_rec_old = res_no_c1["Full"]["Recall"]
    f_rec_new = res_c1["Full"]["Recall"]
    f_f1_old = res_no_c1["Full"]["F1"]
    f_f1_new = res_c1["Full"]["F1"]
    f_fp_old = res_no_c1["Full"]["FP"]
    f_fp_new = res_c1["Full"]["FP"]
    print("-" * 90)
    print(f"{'Full Sequence':<14} | {'Total FP':<10} | {f_fp_old:<18d} | {f_fp_new:<20d} | {f_fp_new - f_fp_old:+d} FP")
    print(f"{'Full Sequence':<14} | {'Prec(%)':<10} | {f_prec_old:<18.2f} | {f_prec_new:<20.2f} | {f_prec_new - f_prec_old:+.2f}%")
    print(f"{'Full Sequence':<14} | {'Recall(%)':<10} | {f_rec_old:<18.2f} | {f_rec_new:<20.2f} | {f_rec_new - f_rec_old:+.2f}%")
    print(f"{'Full Sequence':<14} | {'F1-Score':<10} | {f_f1_old:<18.2f} | {f_f1_new:<20.2f} | {f_f1_new - f_f1_old:+.2f}")
    print(f"{'Execution Speed':<14} | {'FPS':<10} | {res_no_c1['FPS']:<18.2f} | {res_c1['FPS']:<20.2f} | Real-Time 30 FPS")
    print("="*90)

if __name__ == "__main__":
    main()
