# -*- coding: utf-8 -*-
"""
Jetson Nano Benchmark on Completely Unseen Amateur Dataset (am_singles):
========================================================================
- Matches: match_yewtee (1080p, Singapore Yew Tee Community Club)
- Evaluates real-world out-of-domain generalization (Confirmed zero training overlap)
- Strict 30 FPS Camera Queue Emulation (maxsize=3)
- Strict Dual-Penalty Metric:
    * Wrong location (dist > tolerance) counts as FP AND FN
    * Strict Recall (定位成功率) = TP / Total_GT_Visible
- Powered by the unified HierarchicalEdgeTracker (Single Source of Truth)
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

# Import the unified HierarchicalEdgeTracker
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
    frame_interval = 1.0 / 30.0  # Emulate 30 FPS camera ingestion
    
    frame_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stats = {"dropped": 0, "ingested": 0, "processed": 0}
    predictions = {}
    telemetries = {}
    
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
            telemetries[mid_fid] = telemetry
            if telemetry["pred_point"] is not None:
                predictions[mid_fid] = telemetry["pred_point"]
                
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
    
    # Latencies
    latencies = [t["latency_ms"] for t in telemetries.values()]
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    
    return {
        "fps": actual_fps,
        "drop_rate": drop_rate,
        "prec": precision,
        "strict_rec": strict_recall,
        "f1": f1,
        "tp": TP,
        "fp": total_FP,
        "fn": strict_FN,
        "total_visible": total_visible,
        "mean_lat": mean_lat,
        "p95_lat": p95_lat,
        "predictions": predictions
    }


def main():
    print("=" * 105)
    print("   UNSEEN AMATEUR MATCH EVALUATION (Singapore match_yewtee) ON NVIDIA JETSON NANO")
    print("   POWERED BY UNIFIED HierarchicalEdgeTracker (STRICT DUAL-PENALTY)")
    print("=" * 105)
    
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
    print("[INFO] Warmup complete. Commencing evaluation...\n")
    
    # Check candidate directories
    candidate_paths = [
        Path("raw/am_singles"),
        Path("dataset/raw/am_singles"),
        Path("dataset/raw/am_singles/am_singles"),
    ]
    base_raw = None
    for p in candidate_paths:
        if p.exists():
            base_raw = p
            break
            
    if base_raw is None:
        print("[ERROR] Could not find amateur dataset directory!")
        return

    print(f"[INFO] Using amateur dataset at: {base_raw}")
    
    # Focus on Singapore Yew Tee Community Club (1080p)
    clips = [
        {"name": "yewtee_s0", "video": base_raw / "match_yewtee/rally_video/singles0.mp4",
         "csv": base_raw / "match_yewtee/ball_trajectory/singles0_ball.csv", "res": "1920x1080", "tol": 22.5},
        {"name": "yewtee_s1", "video": base_raw / "match_yewtee/rally_video/singles1.mp4",
         "csv": base_raw / "match_yewtee/ball_trajectory/singles1_ball.csv", "res": "1920x1080", "tol": 22.5},
        {"name": "yewtee_s2", "video": base_raw / "match_yewtee/rally_video/singles2.mp4",
         "csv": base_raw / "match_yewtee/ball_trajectory/singles2_ball.csv", "res": "1920x1080", "tol": 22.5},
    ]
    
    results = []
    print(f"{'Clip':<12} | {'Res':<9} | {'FPS':<5} | {'Drop%':<6} | {'Prec%':<6} | {'Strict Rec%':<11} | {'F1':<5} | {'Mean Lat':<9} | {'P95 Lat':<7}")
    print("-" * 95)
    
    tot_tp, tot_fp, tot_fn, tot_gt = 0, 0, 0, 0
    
    for c in clips:
        vpath = str(c["video"])
        cpath = str(c["csv"])
        if not os.path.exists(vpath) or not os.path.exists(cpath):
            print(f"[WARN] Skipping {c['name']}, file not found.")
            continue
            
        res = run_realtime_clip(vpath, cpath, tracker, tolerance=c["tol"])
        results.append((c["name"], c["res"], res))
        
        tot_tp += res["tp"]
        tot_fp += res["fp"]
        tot_fn += res["fn"]
        tot_gt += res["total_visible"]
        
        print(f"{c['name']:<12} | {c['res']:<9} | {res['fps']:>4.1f} | {res['drop_rate']:>5.1f}% | {res['prec']:>5.1f}% | "
              f"{res['strict_rec']:>10.1f}% | {res['f1']:>5.1f} | {res['mean_lat']:>7.1f}ms | {res['p95_lat']:>6.1f}ms")
              
    print("=" * 95)
    if results:
        mean_fps = np.mean([r[2]["fps"] for r in results])
        mean_drop = np.mean([r[2]["drop_rate"] for r in results])
        mean_prec = np.mean([r[2]["prec"] for r in results])
        mean_rec = np.mean([r[2]["strict_rec"] for r in results])
        mean_f1 = np.mean([r[2]["f1"] for r in results])
        mean_lat = np.mean([r[2]["mean_lat"] for r in results])
        
        agg_prec = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else 0.0
        agg_rec = (tot_tp / tot_gt * 100.0) if tot_gt > 0 else 0.0
        agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) > 0 else 0.0
        
        print(f"{'MACRO MEAN':<12} | {'-':<9} | {mean_fps:>4.1f} | {mean_drop:>5.1f}% | {mean_prec:>5.1f}% | "
              f"{mean_rec:>10.1f}% | {mean_f1:>5.1f} | {mean_lat:>7.1f}ms | -")
        print(f"{'AGGREGATE':<12} | {'-':<9} | {'-':<5} | {'-':<6} | {agg_prec:>5.1f}% | "
              f"{agg_rec:>10.1f}% | {agg_f1:>5.1f} | (TP={tot_tp}, FP={tot_fp}, FN={tot_fn}, GT={tot_gt})")
              
        # Save CSV
        out_csv = Path("report_materials/nano_yewtee_evaluation_results.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Clip", "Resolution", "FPS", "Drop_Rate_Pct", "Precision_Pct", "Strict_Recall_Pct", "F1_Score", "Mean_Latency_ms", "P95_Latency_ms", "TP", "FP", "FN", "GT_Visible"])
            for name, rres, r in results:
                writer.writerow([name, rres, f"{r['fps']:.2f}", f"{r['drop_rate']:.2f}", f"{r['prec']:.2f}", f"{r['strict_rec']:.2f}", f"{r['f1']:.2f}", f"{r['mean_lat']:.2f}", f"{r['p95_lat']:.2f}", r["tp"], r["fp"], r["fn"], r["total_visible"]])
        print(f"\n[SUCCESS] Saved results to {out_csv}")


if __name__ == "__main__":
    main()
