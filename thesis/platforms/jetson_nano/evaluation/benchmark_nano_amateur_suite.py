# -*- coding: utf-8 -*-
"""
Jetson Nano Benchmark on Completely Unseen Amateur Dataset (am_singles):
========================================================================
- Matches: match_china2 (720p) & match_yewtee (1080p)
- Out-of-Domain Generalization Test (Confirmed Zero Training Overlap)
- Strict 30 FPS Camera Queue Emulation (maxsize=3)
- Strict Dual-Penalty Metric:
    * Wrong location (dist > 15 px) counts as FP AND FN
    * Strict Recall (定位成功率) = TP / Total_GT_Visible
- Tracks Throughput, Hardware Drop Rate, and End-to-End Decision Latency (Age)
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

def run_realtime_clip(video_path, gt_path, trt_128, trt_256, buf_128, buf_256,
                      t_track=0.18, t_coarse=0.16, t_verify=0.30):
    gt_dict = load_ground_truth(gt_path)
    cap = cv2.VideoCapture(video_path)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / 30.0  # 30 FPS ingestion emulation
    
    frame_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stats = {"dropped": 0, "ingested": 0, "processed": 0}
    predictions = {}
    latencies_local = []
    latencies_rescue = []
    decision_ages = []  # Time elapsed from frame ingestion to prediction completion
    ingest_timestamps = {}
    
    def camera_producer():
        t_next = time.perf_counter()
        f_idx = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            t_ingest = time.perf_counter()
            ingest_timestamps[f_idx] = t_ingest
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
    
    norm_scale = np.float32(1.0 / 255.0)
    resized_slots = [np.zeros((3, 144, 256), dtype=np.float32) for _ in range(3)]
    slot_idx = 0
    num_cached = 0
    
    is_lost = True
    last_x, last_y = 256.0, 144.0
    vx, vy = 0.0, 0.0
    miss_count = 0
    kf = None
    frame_buffer = deque(maxlen=3)
    fid_buffer = deque(maxlen=3)
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    
    t_start_proc = time.perf_counter()
    
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            fid, frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
            
        stats["processed"] += 1
        frame_buffer.append(frame)
        fid_buffer.append(fid)
        
        # Incremental single-frame pre-resize for circular ring buffer
        r_new = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_LINEAR)
        cur_slot = resized_slots[slot_idx % 3]
        cur_slot[:] = np.transpose(r_new, (2, 0, 1)) * norm_scale
        slot_idx += 1
        num_cached += 1
        
        if num_cached < 3:
            continue
            
        target_mid_fid = fid_buffer[1]
        t0 = time.perf_counter()
        pred_pt = None
        
        if not is_lost:
            # Local Tracking Mode (128x128 Crop on native resolution)
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
            pass_vgate = (jump <= 55.0)
            
            if conf >= t_track and pass_vgate:
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
                if miss_count >= 3:
                    is_lost = True
                    kf = None
                    vx, vy = 0.0, 0.0
        else:
            # Two-Stage Coarse-to-Fine Rescue Mode
            buf_256[0, 0:3] = resized_slots[(slot_idx - 3) % 3]
            buf_256[0, 3:6] = resized_slots[(slot_idx - 2) % 3]
            buf_256[0, 6:9] = resized_slots[(slot_idx - 1) % 3]
            
            out_256 = trt_256.infer(buf_256).squeeze()
            gx, gy, conf_coarse = fast_argmax(out_256)
            
            if conf_coarse >= t_coarse:
                cand_x = float(gx) * 2.0
                cand_y = float(gy) * 2.0
                
                v0, _ = robust_fast_crop(frame_buffer[0], cand_x, cand_y, 64)
                v1, _ = robust_fast_crop(frame_buffer[1], cand_x, cand_y, 64)
                v2, _ = robust_fast_crop(frame_buffer[2], cand_x, cand_y, 64)
                
                buf_128[0, 0:3] = np.transpose(v0, (2, 0, 1)) * norm_scale
                buf_128[0, 3:6] = np.transpose(v1, (2, 0, 1)) * norm_scale
                buf_128[0, 6:9] = np.transpose(v2, (2, 0, 1)) * norm_scale
                
                out_ver = trt_128.infer(buf_128).squeeze()
                vx_loc, vy_loc, conf_ver = fast_argmax(out_ver)
                
                if conf_ver >= t_verify:
                    fine_x = cand_x - 64.0 + float(vx_loc)
                    fine_y = cand_y - 64.0 + float(vy_loc)
                    pred_pt = (fine_x * scale_w, fine_y * scale_h)
                    last_x, last_y = fine_x, fine_y
                    is_lost = False
                    miss_count = 0
                    kf = init_kalman_filter(fine_x, fine_y)
                    
            latencies_rescue.append((time.perf_counter() - t0) * 1000.0)
            
        t_finish = time.perf_counter()
        if target_mid_fid in ingest_timestamps:
            # Result age: elapsed time since target middle frame was ingested
            decision_ages.append((t_finish - ingest_timestamps[target_mid_fid]) * 1000.0)
            
        if pred_pt is not None:
            predictions[target_mid_fid] = pred_pt
            
    total_time = time.perf_counter() - t_start_proc
    actual_fps = stats["processed"] / total_time if total_time > 0 else 0.0
    prod_thread.join()
    
    drop_rate = (stats["dropped"] / stats["ingested"] * 100.0) if stats["ingested"] > 0 else 0.0
    
    # -------------------------------------------------------------
    # STRICT EVALUATION METRIC (Dual Penalty for Localization Error)
    # -------------------------------------------------------------
    TP = 0
    FP = 0
    FN = 0
    dist_thresh = 15.0 if h_orig <= 720 else (15.0 * (h_orig / 720.0))  # Scale proportionally for 1080p
    
    total_gt_visible = 0
    for fid in range(total_frames):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gt_x = gt_dict[fid]["x"]
        gt_y = gt_dict[fid]["y"]
        if gvis == 1:
            total_gt_visible += 1
            
        pred = predictions.get(fid, None)
        if pred is not None:
            px, py = pred
            if gvis == 1:
                dist = math.hypot(px - gt_x, py - gt_y)
                if dist <= dist_thresh:
                    TP += 1
                else:
                    # Localization Error: Predicted wrong spot -> FP AND FN!
                    FP += 1
                    FN += 1
            else:
                # Ghost ball detected during dead-ball
                FP += 1
        else:
            if gvis == 1:
                # Complete miss
                FN += 1
                
    prec = (TP / (TP + FP) * 100.0) if (TP + FP) > 0 else 0.0
    rec_strict = (TP / (TP + FN) * 100.0) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec_strict / (prec + rec_strict)) if (prec + rec_strict) > 0 else 0.0
    
    mean_loc = float(np.mean(latencies_local)) if latencies_local else 0.0
    mean_res = float(np.mean(latencies_rescue)) if latencies_rescue else 0.0
    mean_age = float(np.mean(decision_ages)) if decision_ages else 0.0
    p95_age = float(np.percentile(decision_ages, 95)) if decision_ages else 0.0
    
    return {
        "TP": TP, "FP": FP, "FN": FN, "GT_Vis": total_gt_visible,
        "prec": prec, "rec_strict": rec_strict, "f1": f1,
        "fps": actual_fps, "drop_rate": drop_rate,
        "lat_local": mean_loc, "lat_rescue": mean_res,
        "age_mean": mean_age, "age_p95": p95_age,
        "resolution": f"{w_orig}x{h_orig}",
        "total_frames": total_frames,
        "dist_thresh": dist_thresh
    }

def main():
    print("="*105)
    print("   UNSEEN AMATEUR MATCH EVALUATION (am_singles) ON NVIDIA JETSON NANO")
    print("   STRICT DUAL-PENALTY BENCHMARK (Wrong Location = FP + FN)")
    print("="*105)
    
    repo_root = Path("/home/nano/Desktop/tracknet-series-pytorch")
    eng_128_p = str(repo_root / "weights/miniroi_128x128_augmented.engine")
    eng_256_p = str(repo_root / "weights/miniroi_256x144_augmented.engine")
    am_dir = repo_root / "raw/am_singles"
    
    # 7 Labeled clips in am_singles
    test_clips = [
        ("china2_s0", am_dir / "match_china2/rally_video/singles0.mp4", am_dir / "match_china2/ball_trajectory/singles0_ball.csv"),
        ("china2_s1", am_dir / "match_china2/rally_video/singles1.mp4", am_dir / "match_china2/ball_trajectory/singles1_ball.csv"),
        ("china2_s2", am_dir / "match_china2/rally_video/singles2.mp4", am_dir / "match_china2/ball_trajectory/singles2_ball.csv"),
        ("china2_s3", am_dir / "match_china2/rally_video/singles3.mp4", am_dir / "match_china2/ball_trajectory/singles3_ball.csv"),
        ("yewtee_s0", am_dir / "match_yewtee/rally_video/singles0.mp4", am_dir / "match_yewtee/ball_trajectory/singles0_ball.csv"),
        ("yewtee_s1", am_dir / "match_yewtee/rally_video/singles1.mp4", am_dir / "match_yewtee/ball_trajectory/singles1_ball.csv"),
        ("yewtee_s2", am_dir / "match_yewtee/rally_video/singles2.mp4", am_dir / "match_yewtee/ball_trajectory/singles2_ball.csv"),
    ]
    
    cuda_ctx = cuda.Device(0).make_context()
    try:
        trt_128 = FastTRTModel(eng_128_p, cuda_ctx)
        trt_256 = FastTRTModel(eng_256_p, cuda_ctx)
        
        buf_128 = cuda.pagelocked_empty((1, 9, 128, 128), dtype=np.float32)
        buf_256 = cuda.pagelocked_empty((1, 9, 144, 256), dtype=np.float32)
        
        # 10 Warmup inferences
        print("[INFO] Performing 10 warmup inferences on GPU...")
        for _ in range(10):
            _ = trt_128.infer(buf_128)
            _ = trt_256.infer(buf_256)
        print("[INFO] Warmup complete. Commencing unseen amateur evaluation...\n")
        
        print(f"{'Clip':<12} | {'Res':<9} | {'FPS':<5} | {'Drop%':<6} | {'Prec%':<6} | {'Strict Rec%':<11} | {'F1':<5} | {'Mean Age':<8} | {'P95 Age':<8}")
        print("-" * 95)
        
        results = []
        for name, v_p, c_p in test_clips:
            if not v_p.exists() or not c_p.exists():
                print(f"Skipping {name}: file not found ({v_p}).")
                continue
            res = run_realtime_clip(str(v_p), str(c_p), trt_128, trt_256, buf_128, buf_256,
                                    t_track=0.18, t_coarse=0.16, t_verify=0.30)
            res["name"] = name
            results.append(res)
            print(f"{name:<12} | {res['resolution']:<9} | {res['fps']:>4.1f} | {res['drop_rate']:>5.1f}% | {res['prec']:>5.1f}% | {res['rec_strict']:>10.1f}% | {res['f1']:>5.1f} | {res['age_mean']:>6.1f}ms | {res['age_p95']:>6.1f}ms")
            
        print("="*95)
        if results:
            mean_fps = np.mean([r["fps"] for r in results])
            mean_drop = np.mean([r["drop_rate"] for r in results])
            mean_prec = np.mean([r["prec"] for r in results])
            mean_rec = np.mean([r["rec_strict"] for r in results])
            mean_f1 = np.mean([r["f1"] for r in results])
            mean_age = np.mean([r["age_mean"] for r in results])
            
            tot_tp = sum([r["TP"] for r in results])
            tot_fp = sum([r["FP"] for r in results])
            tot_fn = sum([r["FN"] for r in results])
            tot_gt = sum([r["GT_Vis"] for r in results])
            
            agg_prec = tot_tp / (tot_tp + tot_fp) * 100.0 if (tot_tp + tot_fp) > 0 else 0.0
            agg_rec = tot_tp / tot_gt * 100.0 if tot_gt > 0 else 0.0
            agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0.0
            
            print(f"{'MACRO MEAN':<12} | {'-':<9} | {mean_fps:>4.1f} | {mean_drop:>5.1f}% | {mean_prec:>5.1f}% | {mean_rec:>10.1f}% | {mean_f1:>5.1f} | {mean_age:>6.1f}ms | -")
            print(f"{'AGGREGATE':<12} | {'-':<9} | {'-':<4} | {'-':<6} | {agg_prec:>5.1f}% | {agg_rec:>10.1f}% | {agg_f1:>5.1f} | (TP={tot_tp}, FP={tot_fp}, FN={tot_fn}, GT={tot_gt})")
            
            out_csv = repo_root / "report_materials/nano_amateur_evaluation_results.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(str(out_csv), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Clip", "Resolution", "Total_Frames", "GT_Visible", "FPS", "Drop_Rate", "Precision",
                                 "Strict_Recall", "F1", "Mean_Age_ms", "P95_Age_ms", "TP", "FP", "FN"])
                for r in results:
                    writer.writerow([r["name"], r["resolution"], r["total_frames"], r["GT_Vis"], f"{r['fps']:.1f}", f"{r['drop_rate']:.1f}",
                                     f"{r['prec']:.1f}", f"{r['rec_strict']:.1f}", f"{r['f1']:.1f}",
                                     f"{r['age_mean']:.1f}", f"{r['age_p95']:.1f}", r["TP"], r["FP"], r["FN"]])
            print(f"[SUCCESS] Saved results to {out_csv}")
            
    finally:
        cuda_ctx.pop()

if __name__ == "__main__":
    main()
