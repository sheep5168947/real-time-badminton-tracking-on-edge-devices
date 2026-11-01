# -*- coding: utf-8 -*-
"""
Jetson Nano - Base Model Clean A/B Comparison Benchmark
Compares:
  Model A (Old Base): miniroi_128x128.engine (Unaugmented)
  Model B (New Base): miniroi_128x128_augmented.engine (Spatial-Jittering Augmented)

Controls all other variables identically:
  - Shared Rescue Engine: miniroi_256x144.engine
  - Identical tracking logic, identical thresholds, identical video/GT.
  - Evaluates both:
      1. Offline Pure Model Evaluation (All 651 frames, 0 drops, pure AI accuracy)
      2. Real-time 30 FPS Camera Streaming (Hardware drop accounting)
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
import threading
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

def crop_orig_to_128(frame_orig, cx_512, cy_288, radius_512=64):
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

def evaluate_predictions(predictions, gt_dict, total_frames, dist_thresh=15.0):
    TP, FP, FN = 0, 0, 0
    errors = []
    off_center_TP, off_center_total = 0, 0
    
    for fid in range(total_frames):
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

def run_trial(mode_type, model_label, engine_128_path, engine_256_path, video_path, gt_path, thresh_local=0.45, thresh_global=0.35):
    """
    mode_type: 'offline' or 'realtime'
    """
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    gt_dict = load_ground_truth(gt_path)
    
    try:
        trt_128 = FastTRTModel(engine_128_path, cuda_ctx)
        trt_256 = FastTRTModel(engine_256_path, cuda_ctx)
        
        # Pre-allocate pinned memory buffers
        norm_scale = np.float32(1.0 / 255.0)
        buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
        buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        predictions = {}
        processed_fids = set()
        
        is_lost = True
        last_x_512, last_y_288 = 256.0, 144.0
        miss_count = 0
        local_cnt, global_cnt = 0, 0
        latencies = []

        if mode_type == "offline":
            print(f"\n>>> Running OFFLINE Evaluation: {model_label} (Total: {total_video_frames} frames)...", flush=True)
            frame_buffer = deque(maxlen=3)
            fid = 0
            t_start = time.perf_counter()
            
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
                    local_cnt += 1
                    c0 = crop_orig_to_128(frame_buffer[0], last_x_512, last_y_288)
                    c1 = crop_orig_to_128(frame_buffer[1], last_x_512, last_y_288)
                    c2 = crop_orig_to_128(frame_buffer[2], last_x_512, last_y_288)
                    
                    buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * norm_scale
                    buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * norm_scale
                    buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * norm_scale
                    
                    out_128 = trt_128.infer(buf_128).squeeze()
                    lx, ly, conf = fast_argmax(out_128)
                    
                    if conf >= thresh_local:
                        is_lost = False
                        miss_count = 0
                        dx = last_x_512 - 64.0 + float(lx)
                        dy = last_y_288 - 64.0 + float(ly)
                        pred_pt = (dx, dy)
                        last_x_512, last_y_288 = dx, dy
                    else:
                        miss_count += 1
                        if miss_count >= 3:
                            is_lost = True
                else:
                    global_cnt += 1
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
                        pred_pt = (dx, dy)
                        last_x_512, last_y_288 = dx, dy
                
                latencies.append(time.perf_counter() - t0)
                predictions[fid] = pred_pt
                processed_fids.add(fid)
                fid += 1
                
            cap.release()
            t_total = time.perf_counter() - t_start
            metrics = evaluate_predictions(predictions, gt_dict, total_video_frames)
            metrics.update({
                "Mode": "OFFLINE",
                "Model": model_label,
                "TotalFrames": total_video_frames,
                "ProcessedFrames": len(processed_fids),
                "DroppedFrames": 0,
                "DropRate": 0.0,
                "EffectiveFPS": total_video_frames / t_total,
                "AvgLatencyMs": float(np.mean(latencies)) * 1000.0,
                "LocalRatio": (local_cnt / (local_cnt + global_cnt) * 100.0) if (local_cnt + global_cnt) > 0 else 0.0
            })
            return metrics

        elif mode_type == "realtime":
            print(f"\n>>> Running REAL-TIME 30 FPS Ingestion: {model_label}...", flush=True)
            shared = {
                "frame_data": None,
                "frame_id": -1,
                "is_running": True,
                "processed_ids": set(),
                "predictions": {},
                "local_cnt": 0,
                "global_cnt": 0,
                "latencies": []
            }
            frame_lock = threading.Lock()
            
            def ai_worker():
                nonlocal is_lost, last_x_512, last_y_288, miss_count
                cuda_ctx.push()
                try:
                    raw_queue = deque(maxlen=3)
                    last_proc_id = -1
                    
                    while shared["is_running"]:
                        curr_frame = None
                        curr_id = -1
                        with frame_lock:
                            if shared["frame_id"] > last_proc_id:
                                curr_frame = shared["frame_data"]
                                curr_id = shared["frame_id"]
                        
                        if curr_frame is None or curr_id == last_proc_id:
                            time.sleep(0.001)
                            continue
                        
                        last_proc_id = curr_id
                        raw_queue.append(curr_frame)
                        shared["processed_ids"].add(curr_id)
                        
                        if len(raw_queue) < 3:
                            shared["predictions"][curr_id] = None
                            continue
                        
                        t0 = time.perf_counter()
                        pred_pt = None
                        
                        if not is_lost:
                            shared["local_cnt"] += 1
                            c0 = crop_orig_to_128(raw_queue[0], last_x_512, last_y_288)
                            c1 = crop_orig_to_128(raw_queue[1], last_x_512, last_y_288)
                            c2 = crop_orig_to_128(raw_queue[2], last_x_512, last_y_288)
                            
                            buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * norm_scale
                            buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * norm_scale
                            buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * norm_scale
                            
                            out_128 = trt_128.infer(buf_128).squeeze()
                            lx, ly, conf = fast_argmax(out_128)
                            
                            if conf >= thresh_local:
                                is_lost = False
                                miss_count = 0
                                dx = last_x_512 - 64.0 + float(lx)
                                dy = last_y_288 - 64.0 + float(ly)
                                pred_pt = (dx, dy)
                                last_x_512, last_y_288 = dx, dy
                            else:
                                miss_count += 1
                                if miss_count >= 3:
                                    is_lost = True
                        else:
                            shared["global_cnt"] += 1
                            r0 = cv2.resize(raw_queue[0], (256, 144), interpolation=cv2.INTER_LINEAR)
                            r1 = cv2.resize(raw_queue[1], (256, 144), interpolation=cv2.INTER_LINEAR)
                            r2 = cv2.resize(raw_queue[2], (256, 144), interpolation=cv2.INTER_LINEAR)
                            
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
                                pred_pt = (dx, dy)
                                last_x_512, last_y_288 = dx, dy
                        
                        shared["latencies"].append(time.perf_counter() - t0)
                        shared["predictions"][curr_id] = pred_pt
                finally:
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
            
            total_cam = cam_id
            proc_cnt = len(shared["processed_ids"])
            dropped = total_cam - proc_cnt
            drop_rate = (dropped / total_cam * 100.0) if total_cam > 0 else 0.0
            
            metrics = evaluate_predictions(shared["predictions"], gt_dict, total_cam)
            metrics.update({
                "Mode": "REALTIME",
                "Model": model_label,
                "TotalFrames": total_cam,
                "ProcessedFrames": proc_cnt,
                "DroppedFrames": dropped,
                "DropRate": drop_rate,
                "EffectiveFPS": proc_cnt / t_total_cam,
                "AvgLatencyMs": float(np.mean(shared["latencies"])) * 1000.0 if shared["latencies"] else 0.0,
                "LocalRatio": (shared["local_cnt"] / (shared["local_cnt"] + shared["global_cnt"]) * 100.0) if (shared["local_cnt"] + shared["global_cnt"]) > 0 else 0.0
            })
            return metrics
            
    finally:
        cuda_ctx.pop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["offline", "realtime", "both"])
    parser.add_argument("--local_thresh", type=float, default=0.45)
    parser.add_argument("--global_thresh", type=float, default=0.35)
    parser.add_argument("--video", type=str, default="weights/1_01_00.mp4")
    parser.add_argument("--csv", type=str, default="weights/gt.csv")
    args = parser.parse_args()
    
    eng_old_128 = "weights/miniroi_128x128.engine"
    eng_new_128 = "weights/miniroi_128x128_augmented.engine"
    eng_rescue = "weights/miniroi_256x144.engine"
    
    results = []
    modes = ["offline", "realtime"] if args.mode == "both" else [args.mode]
    
    for m in modes:
        print(f"\n=======================================================")
        print(f"  TESTING MODE: {m.upper()} | Local: {args.local_thresh} | Global: {args.global_thresh}")
        print(f"=======================================================")
        
        # Trial 1: Old Base
        res_old = run_trial(m, "Old Base (Unaugmented)", eng_old_128, eng_rescue, args.video, args.csv, args.local_thresh, args.global_thresh)
        results.append(res_old)
        time.sleep(2.0)
        
        # Trial 2: New Base
        res_new = run_trial(m, "New Base (Augmented)", eng_new_128, eng_rescue, args.video, args.csv, args.local_thresh, args.global_thresh)
        results.append(res_new)
        time.sleep(2.0)

    print("\n" + "="*80)
    print("           JETSON NANO BASE MODEL CLEAN A/B COMPARISON RESULTS")
    print("="*80)
    header = f"{'Mode':<10} | {'Model':<24} | {'Prec(%)':<7} | {'Recall(%)':<9} | {'F1':<6} | {'Err(px)':<7} | {'Drop(%)':<7} | {'FPS':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Mode']:<10} | {r['Model']:<24} | {r['Precision']:<7.2f} | {r['Recall']:<9.2f} | {r['F1']:<6.2f} | {r['MeanError']:<7.2f} | {r['DropRate']:<7.2f} | {r['EffectiveFPS']:<6.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
