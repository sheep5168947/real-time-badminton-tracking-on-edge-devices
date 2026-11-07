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
            if telemetry.get("retro_pred") is not None:
                r_fid, r_pt = telemetry["retro_pred"]
                predictions[r_fid] = r_pt
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
    import argparse
    parser = argparse.ArgumentParser(description="Jetson Nano Real-Time Benchmark for match1..match23")
    parser.add_argument("--match_start", type=int, default=1, help="Start match index (e.g. 1)")
    parser.add_argument("--match_end", type=int, default=23, help="End match index (e.g. 23)")
    parser.add_argument("--sample_per_match", type=int, default=0, help="If > 0, test only N clips per match")
    parser.add_argument("--output_csv", type=str, default="report_materials/nano_realtime_full_146videos_results.csv")
    args = parser.parse_args()

    print("=" * 115)
    print("   NVIDIA JETSON NANO REAL-TIME HARDWARE BENCHMARK (match1 ~ match23)")
    print("   POWERED BY UNIFIED HierarchicalEdgeTracker (STRICT DUAL-PENALTY)")
    print("=" * 115)
    
    engine_128 = "weights/miniroi_128x128_augmented.engine" if os.path.exists("weights/miniroi_128x128_augmented.engine") else "weights/miniroi_128x128.engine"
    engine_256 = "weights/miniroi_256x144_augmented.engine" if os.path.exists("weights/miniroi_256x144_augmented.engine") else "weights/miniroi_256x144.engine"
    
    tracker = HierarchicalEdgeTracker(
        engine_128_path=engine_128,
        engine_256_path=engine_256
    )
    
    # Warmup TRT
    print("[INFO] Performing 10 warmup inferences on GPU...")
    d128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
    d256 = np.zeros((1, 9, 144, 256), dtype=np.float32)
    for _ in range(10):
        _ = tracker.trt_128.infer(d128)
        _ = tracker.trt_256.infer(d256)
    print("[INFO] Warmup complete. Commencing evaluation...\n")
    
    candidate_paths = [Path("raw/part1"), Path("dataset/raw/part1")]
    base_raw = None
    for p in candidate_paths:
        if p.exists():
            base_raw = p
            break
            
    if base_raw is None:
        raise FileNotFoundError("Could not find raw/part1 or dataset/raw/part1 directory!")
        
    print(f"[INFO] Using dataset directory: {base_raw}")
    matches = [f"match{i}" for i in range(args.match_start, args.match_end + 1)]
    
    all_results = []
    tot_tp, tot_fp, tot_fn, tot_vis = 0, 0, 0, 0
    tot_frames_all, tot_dropped_all = 0, 0
    
    csv_out = Path(args.output_csv)
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
            if args.sample_per_match > 0:
                v_files = v_files[:args.sample_per_match]
                
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
                tot_tp += res["tp"]
                tot_fp += res["fp"]
                tot_fn += res["fn"]
                tot_vis += res["total_visible"]
                tot_frames_all += res["total_frames"]
                print(f"  {m}/{vf.stem:<10} | FPS: {res['fps']:>4.1f} | Drop: {res['drop_rate']:>4.1f}% | "
                      f"Prec: {res['prec']:>5.1f}% | Strict Rec: {res['strict_rec']:>5.1f}% | F1: {res['f1']:>5.1f}")
                      
    agg_p = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else 0.0
    agg_r = (tot_tp / tot_vis * 100.0) if tot_vis > 0 else 0.0
    agg_f1 = (2 * agg_p * agg_r / (agg_p + agg_r)) if (agg_p + agg_r) > 0 else 0.0
    
    macro_p = float(np.mean([r[2]["prec"] for r in all_results])) if all_results else 0.0
    macro_r = float(np.mean([r[2]["strict_rec"] for r in all_results])) if all_results else 0.0
    macro_f1 = float(np.mean([r[2]["f1"] for r in all_results])) if all_results else 0.0
    macro_fps = float(np.mean([r[2]["fps"] for r in all_results])) if all_results else 0.0
    macro_drop = float(np.mean([r[2]["drop_rate"] for r in all_results])) if all_results else 0.0
    
    print("\n" + "=" * 105)
    print("   BENCHMARK EVALUATION SUMMARY")
    print("=" * 105)
    print(f"Total Clips Tested : {len(all_results)}")
    print(f"Total Frames Tested: {tot_frames_all}")
    print(f"Macro Mean FPS     : {macro_fps:.1f} FPS (Throughput)")
    print(f"Macro Drop Rate    : {macro_drop:.2f}%")
    print(f"Macro Precision    : {macro_p:.1f}%")
    print(f"Macro Strict Recall: {macro_r:.1f}%")
    print(f"Macro F1-Score     : {macro_f1:.1f}")
    print("-" * 105)
    print(f"Aggregate Precision: {agg_p:.1f}%")
    print(f"Aggregate Recall   : {agg_r:.1f}%")
    print(f"Aggregate F1-Score : {agg_f1:.1f}")
    print(f"Total Statistics   : TP={tot_tp}, FP={tot_fp}, FN={tot_fn}, GT_Visible={tot_vis}")
    print("=" * 105)
    print(f"\n[SUCCESS] Completed! Results saved to {csv_out}")


if __name__ == "__main__":
    main()
