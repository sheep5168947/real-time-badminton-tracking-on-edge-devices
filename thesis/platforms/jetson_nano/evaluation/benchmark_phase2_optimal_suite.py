# -*- coding: utf-8 -*-
"""
Phase 2 Jetson Nano Hardware Benchmark:
Runs Real-Time 30 FPS Tracking on Jetson Nano with the Phase-1 Optimal Threshold (T = 0.20).
Features:
- Dual-threaded camera emulation (Producer @ 30 FPS queue + Consumer AI worker)
- TensorRT FP16 execution (miniroi_128x128_augmented.engine + miniroi_256x144_augmented.engine)
- Kinematic Kalman filter prediction
- Velocity-based adaptive crop (128 <-> 160)
- Kinematic Jump Gate (<= 45 px)
- 1-Frame Forward Fill
- Hardware Queue Drop Rate monitoring
- Full Precision, Recall, F1 evaluation across match clips
"""

import sys
import os
import time
import math
import csv
import queue
import threading
from pathlib import Path
from collections import deque
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
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

def robust_fast_crop(frame_orig, cx_512, cy_288, radius_512=64):
    h_orig, w_orig = frame_orig.shape[:2]
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    
    cx_orig = cx_512 * scale_w
    cy_orig = cy_288 * scale_h
    r_orig = int(radius_512 * max(scale_w, scale_h))
    
    x_min, x_max = int(cx_orig - r_orig), int(cx_orig + r_orig)
    y_min, y_max = int(cy_orig - r_orig), int(cy_orig + r_orig)
    
    dim = r_orig * 2
    padded = np.zeros((dim, dim, 3), dtype=np.uint8)
    
    if x_max <= 0 or x_min >= w_orig or y_max <= 0 or y_min >= h_orig:
        return cv2.resize(padded, (128, 128), interpolation=cv2.INTER_LINEAR), (x_min, y_min, x_max, y_max)
        
    vx_min = max(0, x_min)
    vx_max = min(w_orig, x_max)
    vy_min = max(0, y_min)
    vy_max = min(h_orig, y_max)
    
    sub = frame_orig[vy_min:vy_max, vx_min:vx_max]
    off_x = vx_min - x_min
    off_y = vy_min - y_min
    
    target_h = min(sub.shape[0], dim - off_y)
    target_w = min(sub.shape[1], dim - off_x)
    
    if target_h > 0 and target_w > 0:
        padded[off_y : off_y + target_h, off_x : off_x + target_w] = sub[:target_h, :target_w]
        
    return cv2.resize(padded, (128, 128), interpolation=cv2.INTER_LINEAR), (x_min, y_min, x_max, y_max)

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

def run_realtime_clip(video_path, gt_path, engine_128_p, engine_256_p, thresh=0.20):
    gt_dict = load_ground_truth(gt_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / 30.0  # Emulate 30 FPS ingestion
    
    frame_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stats = {"dropped": 0, "ingested": 0, "processed": 0}
    predictions = {}
    latencies_local = []
    latencies_rescue = []
    
    def camera_producer():
        t_next = time.perf_counter()
        f_idx = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            stats["ingested"] += 1
            try:
                frame_queue.put((f_idx, frame), block=False)
            except queue.Full:
                stats["dropped"] += 1
                try:
                    frame_queue.get_nowait()
                    frame_queue.put((f_idx, frame), block=False)
                except Exception:
                    pass
            f_idx += 1
            t_next += frame_interval
            t_sleep = t_next - time.perf_counter()
            if t_sleep > 0:
                time.sleep(t_sleep)
        stop_event.set()
        cap.release()

    prod_thread = threading.Thread(target=camera_producer)
    prod_thread.start()
    
    cuda_ctx = cuda.Device(0).make_context()
    try:
        trt_128 = FastTRTModel(engine_128_p, cuda_ctx)
        trt_256 = FastTRTModel(engine_256_p, cuda_ctx)
        
        norm_scale = np.float32(1.0 / 255.0)
        buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
        buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)
        
        is_lost = True
        last_x, last_y = 256.0, 144.0
        vx, vy = 0.0, 0.0
        miss_count = 0
        kf = None
        frame_buffer = deque(maxlen=3)
        scale_w, scale_h = 2.5, 2.5
        
        t_start_proc = time.perf_counter()
        
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                fid, frame = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
                
            stats["processed"] += 1
            h_f, w_f = frame.shape[:2]
            scale_w = float(w_f) / 512.0
            scale_h = float(h_f) / 288.0
            frame_buffer.append(frame)
            
            if len(frame_buffer) < 3:
                continue
                
            t0 = time.perf_counter()
            pred_pt = None
            
            if not is_lost:
                if kf is not None:
                    pred_k = kf.predict()
                    crop_cx = float(pred_k[0, 0])
                    crop_cy = float(pred_k[1, 0])
                else:
                    crop_cx, crop_cy = last_x, last_y
                    
                speed = math.hypot(vx, vy)
                crop_r = 80 if speed > 22.0 else 64
                scale_factor = (crop_r * 2.0) / 128.0
                
                c0, _ = robust_fast_crop(frame_buffer[0], crop_cx, crop_cy, crop_r)
                c1, _ = robust_fast_crop(frame_buffer[1], crop_cx, crop_cy, crop_r)
                c2, _ = robust_fast_crop(frame_buffer[2], crop_cx, crop_cy, crop_r)
                
                buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * norm_scale
                buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * norm_scale
                buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * norm_scale
                
                out_128 = trt_128.infer(buf_128).squeeze()
                lx, ly, conf = fast_argmax(out_128)
                latencies_local.append((time.perf_counter() - t0) * 1000.0)
                
                dx = crop_cx - float(crop_r) + float(lx) * scale_factor
                dy = crop_cy - float(crop_r) + float(ly) * scale_factor
                jump = math.hypot(dx - last_x, dy - last_y)
                pass_vgate = (jump <= 45.0)
                
                if conf >= thresh and pass_vgate:
                    miss_count = 0
                    vx = dx - last_x
                    vy = dy - last_y
                    pred_pt = (dx * scale_w, dy * scale_h)
                    last_x, last_y = dx, dy
                    if kf is None:
                        kf = init_kalman_filter(dx, dy)
                    else:
                        kf.correct(np.array([[dx], [dy]], dtype=np.float32))
                else:
                    miss_count += 1
                    if miss_count <= 1:
                        if kf is not None:
                            pred_pt = (crop_cx * scale_w, crop_cy * scale_h)
                            last_x, last_y = crop_cx, crop_cy
                        else:
                            pred_pt = ((last_x + vx) * scale_w, (last_y + vy) * scale_h)
                            last_x, last_y = last_x + vx, last_y + vy
                    if miss_count >= 3:
                        is_lost = True
                        kf = None
                        vx, vy = 0.0, 0.0
            else:
                # Global rescue 256
                r0 = cv2.resize(frame_buffer[0], (256, 144))
                r1 = cv2.resize(frame_buffer[1], (256, 144))
                r2 = cv2.resize(frame_buffer[2], (256, 144))
                
                buf_256[0, 0:3] = np.transpose(r0, (2, 0, 1)) * norm_scale
                buf_256[0, 3:6] = np.transpose(r1, (2, 0, 1)) * norm_scale
                buf_256[0, 6:9] = np.transpose(r2, (2, 0, 1)) * norm_scale
                
                out_256 = trt_256.infer(buf_256).squeeze()
                gx, gy, conf = fast_argmax(out_256)
                latencies_rescue.append((time.perf_counter() - t0) * 1000.0)
                
                if conf >= thresh:
                    dx = float(gx) * 2.0
                    dy = float(gy) * 2.0
                    pred_pt = (dx * scale_w, dy * scale_h)
                    last_x, last_y = dx, dy
                    is_lost = False
                    miss_count = 0
                    kf = init_kalman_filter(dx, dy)
                    
            if pred_pt is not None:
                predictions[fid] = pred_pt
                
        total_time = time.perf_counter() - t_start_proc
        actual_fps = stats["processed"] / total_time if total_time > 0 else 0.0
    finally:
        cuda_ctx.pop()
        prod_thread.join()
        
    drop_rate = (stats["dropped"] / stats["ingested"] * 100.0) if stats["ingested"] > 0 else 0.0
    
    # Calculate Precision and Recall
    TP, FP, FN = 0, 0, 0
    dist_thresh = 15.0
    for fid in range(total_frames):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gt_x = gt_dict[fid]["x"]
        gt_y = gt_dict[fid]["y"]
        pred = predictions.get(fid, None)
        if pred is not None:
            px, py = pred
            if gvis == 1:
                dist = math.hypot(px - gt_x, py - gt_y)
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
    mean_lat_local = float(np.mean(latencies_local)) if latencies_local else 0.0
    mean_lat_rescue = float(np.mean(latencies_rescue)) if latencies_rescue else 0.0
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "prec": prec, "rec": rec, "f1": f1,
        "fps": actual_fps, "drop_rate": drop_rate,
        "lat_local": mean_lat_local, "lat_rescue": mean_lat_rescue,
        "ingested": stats["ingested"], "processed": stats["processed"], "dropped": stats["dropped"]
    }

def main():
    print("="*85)
    print("   PHASE 2: JETSON NANO HARDWARE REAL-TIME 30 FPS BENCHMARK (T = 0.20)")
    print("="*85)
    
    repo_root = Path("/home/nano/Desktop/tracknet-series-pytorch")
    eng_128 = str(repo_root / "weights/miniroi_128x128_augmented.engine")
    eng_256 = str(repo_root / "weights/miniroi_256x144_augmented.engine")
    
    # Test suite covering multiple matches across the dataset
    test_clips = [
        ("match1_1_01_00", repo_root / "raw/part1/match1/video/1_01_00.mp4", repo_root / "raw/part1/match1/csv/1_01_00_ball.csv"),
        ("match3_1_01_00", repo_root / "raw/part1/match3/video/1_01_00.mp4", repo_root / "raw/part1/match3/csv/1_01_00_ball.csv"),
        ("match10_1_03_01", repo_root / "raw/part1/match10/video/1_03_01.mp4", repo_root / "raw/part1/match10/csv/1_03_01_ball.csv"),
        ("match12_2_01_01", repo_root / "raw/part1/match12/video/2_01_01.mp4", repo_root / "raw/part1/match12/csv/2_01_01_ball.csv"),
        ("match13_1_09_10", repo_root / "raw/part1/match13/video/1_09_10.mp4", repo_root / "raw/part1/match13/csv/1_09_10_ball.csv"),
        ("match19_1_01_03", repo_root / "raw/part1/match19/video/1_01_03.mp4", repo_root / "raw/part1/match19/csv/1_01_03_ball.csv"),
    ]
    
    results = []
    print(f"{'Clip':<18} | {'FPS':<6} | {'Drop%':<7} | {'Local(ms)':<9} | {'Rescue(ms)':<10} | {'Prec%':<7} | {'Rec%':<7} | {'F1':<6}")
    print("-"*85)
    
    for name, v_p, c_p in test_clips:
        if not v_p.exists() or not c_p.exists():
            print(f"Skipping {name}: file not found.")
            continue
        res = run_realtime_clip(str(v_p), str(c_p), eng_128, eng_256, thresh=0.20)
        res["name"] = name
        results.append(res)
        print(f"{name:<18} | {res['fps']:>5.1f}  | {res['drop_rate']:>5.1f}%  | {res['lat_local']:>7.2f}ms  | {res['lat_rescue']:>8.2f}ms  | {res['prec']:>5.1f}% | {res['rec']:>5.1f}% | {res['f1']:>5.1f}")
        
    print("="*85)
    if results:
        mean_fps = np.mean([r["fps"] for r in results])
        mean_drop = np.mean([r["drop_rate"] for r in results])
        mean_prec = np.mean([r["prec"] for r in results])
        mean_rec = np.mean([r["rec"] for r in results])
        mean_f1 = np.mean([r["f1"] for r in results])
        mean_loc = np.mean([r["lat_local"] for r in results])
        mean_res = np.mean([r["lat_rescue"] for r in results])
        
        print(f"{'SUITE AVERAGE':<18} | {mean_fps:>5.1f}  | {mean_drop:>5.1f}%  | {mean_loc:>7.2f}ms  | {mean_res:>8.2f}ms  | {mean_prec:>5.1f}% | {mean_rec:>5.1f}% | {mean_f1:>5.1f}")
        
        # Save summary
        out_csv = repo_root / "phase2_nano_realtime_summary.csv"
        with open(str(out_csv), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Clip", "FPS", "Drop_Rate", "Local_Lat_ms", "Rescue_Lat_ms", "Prec", "Rec", "F1", "TP", "FP", "FN"])
            for r in results:
                writer.writerow([r["name"], f"{r['fps']:.1f}", f"{r['drop_rate']:.1f}", f"{r['lat_local']:.2f}", f"{r['lat_rescue']:.2f}",
                                 f"{r['prec']:.1f}", f"{r['rec']:.1f}", f"{r['f1']:.1f}", r["TP"], r["FP"], r["FN"]])

if __name__ == "__main__":
    main()
