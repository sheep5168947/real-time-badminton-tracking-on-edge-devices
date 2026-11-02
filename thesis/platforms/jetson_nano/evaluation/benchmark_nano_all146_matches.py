# -*- coding: utf-8 -*-
"""
Jetson Nano Master Thesis Final Benchmark:
Comprehensive 146-Video Real-Time 30 FPS Evaluation across match1 to match23.
=============================================================================
Powered by Unified HierarchicalEdgeTracker (Single Source of Truth)
Strict Dual-Penalty Metric: (Wrong Location = FP + FN)
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

# Import unified HierarchicalEdgeTracker
try:
    from thesis.platforms.jetson_nano.tracker import HierarchicalEdgeTracker
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
    from thesis.platforms.jetson_nano.tracker import HierarchicalEdgeTracker


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


def run_realtime_clip(video_path, gt_path, tracker, tolerance=15.0):
    tracker.reset()
    gt_dict = load_ground_truth(gt_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / 30.0  # Emulate 30 FPS camera ingest
    
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
    
    t_start_proc = time.perf_counter()
    
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            fid, frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
            
        stats["processed"] += 1
        telemetry = tracker.process_frame(frame, frame_id=fid)
        
        if telemetry is not None:
            mid_fid = telemetry["frame_id"]
            if telemetry["pred_point"] is not None:
                predictions[mid_fid] = telemetry["pred_point"]
            if "LOCAL" in telemetry["mode"] or "STATIONARY" in telemetry["mode"]:
                latencies_local.append(telemetry["latency_ms"])
            else:
                latencies_rescue.append(telemetry["latency_ms"])
                
    total_time = time.perf_counter() - t_start_proc
    actual_fps = stats["processed"] / total_time if total_time > 0 else 0.0
    prod_thread.join()
    
    drop_rate = (stats["dropped"] / stats["ingested"] * 100.0) if stats["ingested"] > 0 else 0.0
    
    # Strict Dual-Penalty Metric (Wrong location = FP + FN)
    TP = 0
    FP_wrong_loc = 0
    FP_ghost = 0
    FN_miss = 0
    total_visible = 0
    
    for fid in range(total_frames):
        if fid not in gt_dict:
            continue
        gvis = gt_dict[fid]["vis"]
        gx = gt_dict[fid]["x"]
        gy = gt_dict[fid]["y"]
        pred = predictions.get(fid, None)
        
        if gvis == 1:
            total_visible += 1
            if pred is not None:
                dist = math.hypot(pred[0] - gx, pred[1] - gy)
                if dist <= tolerance:
                    TP += 1
                else:
                    FP_wrong_loc += 1
            else:
                FN_miss += 1
        else:
            if pred is not None:
                FP_ghost += 1
                
    total_FP = FP_wrong_loc + FP_ghost
    strict_FN = FN_miss + FP_wrong_loc
    
    precision = (TP / (TP + total_FP) * 100.0) if (TP + total_FP) > 0 else 0.0
    strict_recall = (TP / total_visible * 100.0) if total_visible > 0 else 0.0
    f1 = (2 * precision * strict_recall / (precision + strict_recall)) if (precision + strict_recall) > 0 else 0.0
    
    avg_loc_lat = float(np.mean(latencies_local)) if latencies_local else 0.0
    avg_rec_lat = float(np.mean(latencies_rescue)) if latencies_rescue else 0.0
    
    return {
        "fps": actual_fps,
        "drop_rate": drop_rate,
        "local_lat": avg_loc_lat,
        "rescue_lat": avg_rec_lat,
        "prec": precision,
        "strict_rec": strict_recall,
        "f1": f1,
        "tp": TP,
        "fp": total_FP,
        "fn": strict_FN,
        "total_visible": total_visible,
        "total_frames": total_frames
    }


def main():
    print("=" * 115)
    print("   NVIDIA JETSON NANO 146-VIDEO REAL-TIME HARDWARE BENCHMARK (match1 ~ match23)")
    print("   POWERED BY UNIFIED HierarchicalEdgeTracker (STRICT DUAL-PENALTY)")
    print("=" * 115)
    
    engine_128 = "weights/miniroi_128x128.engine"
    engine_256 = "weights/miniroi_256x144.engine"
    
    tracker = HierarchicalEdgeTracker(
        engine_128_path=engine_128,
        engine_256_path=engine_256,
        t_track=0.18,
        t_coarse=0.16,
        t_verify=0.28,
        adaptive_rescue=True
    )
    
    # Warmup TRT
    print("[INFO] Performing 10 warmup inferences on GPU...")
    d128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
    d256 = np.zeros((1, 9, 144, 256), dtype=np.float32)
    for _ in range(10):
        _ = tracker.trt_128.infer(d128)
        _ = tracker.trt_256.infer(d256)
    print("[INFO] Warmup complete. Commencing 146-video benchmark...\n")
    
    base_raw = Path("dataset/raw/part1")
    matches = [f"match{i}" for i in range(1, 24)]
    
    all_results = []
    
    csv_out = Path("report_materials/nano_realtime_full_146videos_results.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "Match", "Clip", "Total_Frames", "FPS", "Drop_Rate_Pct",
            "Local_Lat_ms", "Rescue_Lat_ms", "Precision_Pct", "Recall_Pct",
            "F1_Score", "TP", "FP", "FN", "GT_Visible"
        ])
        
        for m in matches:
            m_dir = base_raw / m
            v_dir = m_dir / "video"
            c_dir = m_dir / "csv"
            if not v_dir.exists() or not c_dir.exists():
                continue
                
            v_files = sorted(list(v_dir.glob("*.mp4")))
            print(f"\n>>> Running {m} ({len(v_files)} clips) ...")
            
            for vf in v_files:
                cf = c_dir / f"{vf.stem}_ball.csv"
                if not cf.exists():
                    continue
                    
                res = run_realtime_clip(str(vf), str(cf), tracker, tolerance=15.0)
                writer.writerow([
                    m, vf.stem, res["total_frames"], f"{res['fps']:.1f}", f"{res['drop_rate']:.1f}",
                    f"{res['local_lat']:.2f}", f"{res['rescue_lat']:.2f}", f"{res['prec']:.1f}",
                    f"{res['strict_rec']:.1f}", f"{res['f1']:.1f}", res["tp"], res["fp"], res["fn"], res["total_visible"]
                ])
                f_out.flush()
                all_results.append((m, vf.stem, res))
                print(f"  {m}/{vf.stem:<10} | FPS: {res['fps']:>4.1f} | Drop: {res['drop_rate']:>4.1f}% | "
                      f"Prec: {res['prec']:>5.1f}% | Strict Rec: {res['strict_rec']:>5.1f}% | F1: {res['f1']:>5.1f}")
                      
    print(f"\n[SUCCESS] Completed all {len(all_results)} videos! Saved to {csv_out}")


if __name__ == "__main__":
    main()
