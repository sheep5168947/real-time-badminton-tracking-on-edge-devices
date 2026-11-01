# -*- coding: utf-8 -*-
"""
Jetson Nano - Scheme B: Unseen Amateur Dataset Benchmark (am_singles)
Evaluates 7 unseen amateur match clips across two venues (match_china2 & match_yewtee):
  Compares:
    1. Baseline: Pure Base (Static Crop, No KF, No Adaptive, No Fill)
    2. Proposed: Full Optimized System (KF-Pred + Velocity-Gate + Adaptive-ROI 160 + 1F-Fill + 0.35 Dual Thresh)

Outputs:
  - Per-clip metrics (Precision, Recall, F1, TP, FP, FN, FPS)
  - Overall aggregate dataset metrics across all 2,159 frames!
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
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    
    cx_orig = cx_512 * scale_w
    cy_orig = cy_288 * scale_h
    r_orig_w = int(radius_512 * scale_w)
    r_orig_h = int(radius_512 * scale_h)
    r_orig = max(r_orig_w, r_orig_h)
    
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

def evaluate_clip(predictions, gt_dict, total_frames, w_orig, h_orig):
    dist_thresh = 15.0 * (float(w_orig) / 1280.0)
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    
    TP, FP, FN = 0, 0, 0
    errors = []
    
    for fid in range(total_frames):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gt_x = gt_dict[fid]["x"]
        gt_y = gt_dict[fid]["y"]
        pred = predictions.get(fid, None)

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
    mean_err = float(np.mean(errors)) if errors else 0.0
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": prec, "Recall": rec, "F1": f1,
        "MeanError": mean_err
    }

def run_single_clip(video_path, csv_path, use_proposed, trt_128, trt_256, buf_128, buf_256,
                     thresh_local=0.35, thresh_global=0.35):
    
    cap = cv2.VideoCapture(video_path)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_dict = load_ground_truth(csv_path)
    
    norm_scale = np.float32(1.0 / 255.0)
    predictions = {}
    
    is_lost = True
    last_x_512, last_y_288 = 256.0, 144.0
    vx, vy = 0.0, 0.0
    miss_count = 0
    kf = None
    
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
        
        if not is_lost:
            # Determine crop center
            if use_proposed and kf is not None:
                predicted = kf.predict()
                crop_cx = float(predicted[0, 0])
                crop_cy = float(predicted[1, 0])
            else:
                crop_cx, crop_cy = last_x_512, last_y_288
            
            # Determine crop radius
            crop_r = 64
            if use_proposed:
                speed = math.hypot(vx, vy)
                if speed > 22.0:
                    crop_r = 80
            
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
            
            pass_vgate = True
            if use_proposed:
                pass_vgate = (jump_dist <= 45.0)
            
            if conf >= thresh_local and pass_vgate:
                is_lost = False
                miss_count = 0
                vx = dx - last_x_512
                vy = dy - last_y_288
                pred_pt = (dx, dy)
                last_x_512, last_y_288 = dx, dy
                if use_proposed:
                    if kf is None:
                        kf = init_kalman_filter(dx, dy)
                    else:
                        kf.correct(np.array([[dx], [dy]], dtype=np.float32))
            else:
                miss_count += 1
                if use_proposed and miss_count <= 1:
                    # 1-Frame Forward Fill
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
                if use_proposed:
                    kf = init_kalman_filter(dx, dy)
        
        latencies.append(time.perf_counter() - t0)
        predictions[fid] = pred_pt
        fid += 1
        
    cap.release()
    t_total = time.perf_counter() - t_start
    metrics = evaluate_clip(predictions, gt_dict, total_video_frames, w_orig, h_orig)
    metrics["FPS"] = total_video_frames / t_total
    metrics["TotalFrames"] = total_video_frames
    return metrics

def main():
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    
    eng_128 = "weights/miniroi_128x128_augmented.engine"
    eng_256 = "weights/miniroi_256x144.engine"
    
    trt_128 = FastTRTModel(eng_128, cuda_ctx)
    trt_256 = FastTRTModel(eng_256, cuda_ctx)
    buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
    buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)
    
    clips = [
        ("china2_s0", "raw/am_singles/match_china2/rally_video/singles0.mp4", "raw/am_singles/match_china2/ball_trajectory/singles0_ball.csv"),
        ("china2_s1", "raw/am_singles/match_china2/rally_video/singles1.mp4", "raw/am_singles/match_china2/ball_trajectory/singles1_ball.csv"),
        ("china2_s2", "raw/am_singles/match_china2/rally_video/singles2.mp4", "raw/am_singles/match_china2/ball_trajectory/singles2_ball.csv"),
        ("china2_s3", "raw/am_singles/match_china2/rally_video/singles3.mp4", "raw/am_singles/match_china2/ball_trajectory/singles3_ball.csv"),
        ("yewtee_s0", "raw/am_singles/match_yewtee/rally_video/singles0.mp4", "raw/am_singles/match_yewtee/ball_trajectory/singles0_ball.csv"),
        ("yewtee_s1", "raw/am_singles/match_yewtee/rally_video/singles1.mp4", "raw/am_singles/match_yewtee/ball_trajectory/singles1_ball.csv"),
        ("yewtee_s2", "raw/am_singles/match_yewtee/rally_video/singles2.mp4", "raw/am_singles/match_yewtee/ball_trajectory/singles2_ball.csv"),
    ]
    
    print("="*95)
    print("     SCHEME B: FULL UNSEEN AMATEUR DATASET BENCHMARK (7 MATCH CLIPS)")
    print("="*95)
    
    results_base = []
    results_prop = []
    
    for label, v_path, c_path in clips:
        print(f"\n>>> Benchmarking Clip: {label} ({v_path})...", flush=True)
        
        # 1. Baseline
        m_base = run_single_clip(v_path, c_path, False, trt_128, trt_256, buf_128, buf_256)
        m_base["Clip"] = label
        results_base.append(m_base)
        print(f"    [Baseline] --> Prec: {m_base['Precision']:5.2f}% | Recall: {m_base['Recall']:5.2f}% | F1: {m_base['F1']:5.2f} | FPS: {m_base['FPS']:.1f}")
        
        # 2. Proposed
        m_prop = run_single_clip(v_path, c_path, True, trt_128, trt_256, buf_128, buf_256)
        m_prop["Clip"] = label
        results_prop.append(m_prop)
        print(f"    [Proposed] --> Prec: {m_prop['Precision']:5.2f}% | Recall: {m_prop['Recall']:5.2f}% | F1: {m_prop['F1']:5.2f} | FPS: {m_prop['FPS']:.1f}")
        time.sleep(0.5)

    # Summary Table
    print("\n" + "="*95)
    print("           UNSEEN AMATEUR MATCH EVALUATION (BASELINE vs PROPOSED)")
    print("="*95)
    header = f"{'Clip Label':<12} | {'Base Prec':<10} | {'Prop Prec':<10} | {'Base Rec':<10} | {'Prop Rec':<10} | {'Base F1':<8} | {'Prop F1':<8} | {'FPS':<6}"
    print(header)
    print("-" * 95)
    
    tot_tp_b, tot_fp_b, tot_fn_b = 0, 0, 0
    tot_tp_p, tot_fp_p, tot_fn_p = 0, 0, 0
    tot_frames = 0
    fps_list = []
    
    for b, p in zip(results_base, results_prop):
        print(f"{b['Clip']:<12} | {b['Precision']:<10.2f} | {p['Precision']:<10.2f} | {b['Recall']:<10.2f} | {p['Recall']:<10.2f} | {b['F1']:<8.2f} | {p['F1']:<8.2f} | {p['FPS']:<6.1f}")
        tot_tp_b += b["TP"]; tot_fp_b += b["FP"]; tot_fn_b += b["FN"]
        tot_tp_p += p["TP"]; tot_fp_p += p["FP"]; tot_fn_p += p["FN"]
        tot_frames += p["TotalFrames"]
        fps_list.append(p["FPS"])
        
    print("-" * 95)
    # Global aggregates
    g_prec_b = (tot_tp_b / (tot_tp_b + tot_fp_b) * 100.0) if (tot_tp_b + tot_fp_b) > 0 else 0.0
    g_rec_b = (tot_tp_b / (tot_tp_b + tot_fn_b) * 100.0) if (tot_tp_b + tot_fn_b) > 0 else 0.0
    g_f1_b = (2 * g_prec_b * g_rec_b / (g_prec_b + g_rec_b)) if (g_prec_b + g_rec_b) > 0 else 0.0
    
    g_prec_p = (tot_tp_p / (tot_tp_p + tot_fp_p) * 100.0) if (tot_tp_p + tot_fp_p) > 0 else 0.0
    g_rec_p = (tot_tp_p / (tot_tp_p + tot_fn_p) * 100.0) if (tot_tp_p + tot_fn_p) > 0 else 0.0
    g_f1_p = (2 * g_prec_p * g_rec_p / (g_prec_p + g_rec_p)) if (g_prec_p + g_rec_p) > 0 else 0.0
    
    avg_fps = float(np.mean(fps_list))
    print(f"{'OVERALL (ALL)':<12} | {g_prec_b:<10.2f} | {g_prec_p:<10.2f} | {g_rec_b:<10.2f} | {g_rec_p:<10.2f} | {g_f1_b:<8.2f} | {g_f1_p:<8.2f} | {avg_fps:<6.1f}")
    print("="*95)
    print(f"Total Evaluated Frames: {tot_frames} frames across 7 amateur matches.")
    print(f"Recall Improvement: {g_rec_b:.2f}% --> {g_rec_p:.2f}% ({g_rec_p - g_rec_b:+.2f}%)")
    print(f"Precision Improvement: {g_prec_b:.2f}% --> {g_prec_p:.2f}% ({g_prec_p - g_prec_b:+.2f}%)")
    print(f"F1-Score Improvement: {g_f1_b:.2f} --> {g_f1_p:.2f} ({g_f1_p - g_f1_b:+.2f})")
    print("="*95)
    
    cuda_ctx.pop()

if __name__ == "__main__":
    main()
