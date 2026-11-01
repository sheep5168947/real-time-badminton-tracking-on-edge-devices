# -*- coding: utf-8 -*-
"""
Jetson Nano - Optimal Threshold Bracket & Bisection Search (夾擠法尋找最佳門檻)
Uses Config 5 (KF-Pred + Velocity-Gate + Adaptive-ROI + Trajectory-Fill 1F)
Evaluates thresholds across [0.28, 0.52] to find the mathematical peak of F1-Score & Recall.
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

def evaluate_predictions(predictions, gt_dict, total_frames, dist_thresh=15.0, start_frame=330):
    TP, FP, FN = 0, 0, 0
    errors = []
    
    for fid in range(start_frame, total_frames):
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

def run_single_eval(thresh_local, thresh_global, trt_128, trt_256, buf_128, buf_256, video_path, gt_dict):
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    norm_scale = np.float32(1.0 / 255.0)
    
    predictions = {}
    is_lost = True
    last_x_512, last_y_288 = 256.0, 144.0
    vx, vy = 0.0, 0.0
    miss_count = 0
    kf = None
    
    frame_buffer = deque(maxlen=3)
    fid = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
        
        if len(frame_buffer) < 3:
            predictions[fid] = None
            fid += 1
            continue
        
        pred_pt = None
        
        if not is_lost:
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
                is_lost = False
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
                    is_lost = True
                    kf = None
                    vx, vy = 0.0, 0.0
        else:
            r0 = cv2.resize(frame_buffer[0], (256, 144), interpolation=cv2.INTER_LINEAR)
            r1 = cv2.resize(frame_buffer[1], (256, 144), interpolation=cv2.INTER_LINEAR)
            r2 = cv2.resize(frame_buffer[2], (256, 144), interpolation=cv2.INTER_LINEAR)
            
            buf_256[0, 0:3] = np.transpose(r0, (2, 0, 1)) * norm_scale
            buf_256[0, 3:6] = np.transpose(r1, (2, 0, 1)) * norm_scale
            buf_256[0, 6:9] = np.transpose(r2, (2, 0, 1)) * norm_scale
            
            out_256 = trt_256.infer(buf_256).squeeze()
            gx, gy, conf = fast_argmax(out_256)
            
            if conf >= thresh_global:
                is_lost = False
                miss_count = 0
                dx = float(gx) * 2.0
                dy = float(gy) * 2.0
                vx, vy = 0.0, 0.0
                pred_pt = (dx, dy)
                last_x_512, last_y_288 = dx, dy
                kf = init_kalman_filter(dx, dy)
        
        predictions[fid] = pred_pt
        fid += 1
        
    cap.release()
    metrics = evaluate_predictions(predictions, gt_dict, total_video_frames, start_frame=330)
    return metrics

def main():
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    video_path = "weights/1_01_00.mp4"
    gt_path = "weights/gt.csv"
    gt_dict = load_ground_truth(gt_path)
    
    eng_128 = "weights/miniroi_128x128_augmented.engine"
    eng_256 = "weights/miniroi_256x144.engine"
    
    trt_128 = FastTRTModel(eng_128, cuda_ctx)
    trt_256 = FastTRTModel(eng_256, cuda_ctx)
    buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
    buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)
    
    print("="*85)
    print("    THRESHOLD BRACKET SEARCH (夾擠法): OPTIMIZING LOCAL CONFIDENCE")
    print("="*85)
    
    # 1. Coarse Grid Bracket: [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    coarse_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    global_thresh = 0.28
    
    results = []
    for th in coarse_thresholds:
        t0 = time.time()
        m = run_single_eval(th, global_thresh, trt_128, trt_256, buf_128, buf_256, video_path, gt_dict)
        elapsed = time.time() - t0
        m["Threshold"] = th
        results.append(m)
        print(f"Threshold: {th:.2f} | Rally Recall: {m['Recall']:6.2f}% | Precision: {m['Precision']:6.2f}% | F1: {m['F1']:6.2f} | TP: {m['TP']:3d} | FN: {m['FN']:3d} | FP: {m['FP']:3d} ({elapsed:.1f}s)")
    
    # Find best coarse threshold
    best_coarse = max(results, key=lambda x: x["F1"])
    best_th = best_coarse["Threshold"]
    print(f"\n>>> Coarse Best: Threshold = {best_th:.2f} (F1 = {best_coarse['F1']:.2f}, Recall = {best_coarse['Recall']:.2f}%)")
    
    # 2. Fine Bisection / Bracket around best_th (step = 0.02)
    fine_thresholds = [best_th - 0.03, best_th - 0.015, best_th + 0.015, best_th + 0.03]
    fine_thresholds = [round(x, 3) for x in fine_thresholds if 0.20 <= x <= 0.55 and x not in coarse_thresholds]
    
    print("\n>>> Running Fine Bracket Search around best threshold...")
    for th in fine_thresholds:
        t0 = time.time()
        m = run_single_eval(th, global_thresh, trt_128, trt_256, buf_128, buf_256, video_path, gt_dict)
        elapsed = time.time() - t0
        m["Threshold"] = th
        results.append(m)
        print(f"Threshold: {th:.3f} | Rally Recall: {m['Recall']:6.2f}% | Precision: {m['Precision']:6.2f}% | F1: {m['F1']:6.2f} | TP: {m['TP']:3d} | FN: {m['FN']:3d} | FP: {m['FP']:3d} ({elapsed:.1f}s)")
        
    results.sort(key=lambda x: x["Threshold"])
    
    print("\n" + "="*85)
    print("                     FINAL THRESHOLD SWEEP SUMMARY")
    print("="*85)
    header = f"{'Threshold':<10} | {'Recall(%)':<10} | {'Precision(%)':<13} | {'F1-Score':<9} | {'TP':<4} | {'FP':<4} | {'FN':<4}"
    print(header)
    print("-" * len(header))
    for r in results:
        star = " *" if r == max(results, key=lambda x: x["F1"]) else ""
        print(f"{r['Threshold']:<10.3f} | {r['Recall']:<10.2f} | {r['Precision']:<13.2f} | {r['F1']:<9.2f} | {r['TP']:<4} | {r['FP']:<4} | {r['FN']:<4}{star}")
    print("="*85)
    
    best_final = max(results, key=lambda x: x["F1"])
    print(f"\nOPTIMAL THRESHOLD FOUND: {best_final['Threshold']:.3f} --> Max F1: {best_final['F1']:.2f}%, Recall: {best_final['Recall']:.2f}%, Precision: {best_final['Precision']:.2f}%")
    cuda_ctx.pop()

if __name__ == "__main__":
    main()
