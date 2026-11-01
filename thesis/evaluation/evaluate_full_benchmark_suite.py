# -*- coding: utf-8 -*-
"""
Comprehensive Dataset-Wide Evaluation Suite for Edge Badminton Tracking.
Evaluates across all unseen validation clips in TrackNet benchmark:
- Computes Mean and Median Precision, Recall, F1, and Localization Error.
- Evaluates both Aggregate (dataset-wide sum) and Per-Clip distributions.
- Compares:
    1. Baseline Tracker (Static 128, No KF, No Velocity Gate, No Adaptive, No Fill)
    2. Proposed System (KF Prediction, Adaptive 160 ROI, Velocity Jump Gate <=45px, 1-Frame Forward Fill, 256 Fast Rescue)
"""

import sys
import os
from pathlib import Path
from collections import deque
import csv
import math
import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thesis.models.miniroi_unet import MiniROIUNet

def init_kalman_filter(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf

def fast_crop_orig(frame_orig, cx_512, cy_288, radius_512=64):
    h_orig, w_orig = frame_orig.shape[:2]
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    
    cx_orig = cx_512 * scale_w
    cy_orig = cy_288 * scale_h
    r_orig = int(radius_512 * max(scale_w, scale_h))
    
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
        
    return cv2.resize(sub, (128, 128), interpolation=cv2.INTER_LINEAR), (x_min, y_min, x_max, y_max)

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

def evaluate_clip(model, video_path, csv_path, device, mode="proposed", thresh=0.35):
    gt_dict = load_ground_truth(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    is_lost = True
    last_x, last_y = 256.0, 144.0
    vx, vy = 0.0, 0.0
    miss_count = 0
    kf = None
    
    frame_buffer = deque(maxlen=3)
    norm_scale = 1.0 / 255.0
    predictions = {}
    
    with torch.no_grad():
        for fid in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            if len(frame_buffer) < 3:
                continue
                
            pred_pt = None
            
            if mode == "baseline":
                # Baseline: Static 128, No KF, No Jump Gate, No Adaptive, No Fill
                if not is_lost:
                    c0, _ = fast_crop_orig(frame_buffer[0], last_x, last_y, 64)
                    c1, _ = fast_crop_orig(frame_buffer[1], last_x, last_y, 64)
                    c2, _ = fast_crop_orig(frame_buffer[2], last_x, last_y, 64)
                    inp = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    lx, ly = int(idx[-1]), int(idx[-2])
                    
                    if conf >= thresh:
                        dx = last_x - 64.0 + float(lx)
                        dy = last_y - 64.0 + float(ly)
                        pred_pt = (dx * scale_w, dy * scale_h)
                        last_x, last_y = dx, dy
                        miss_count = 0
                    else:
                        miss_count += 1
                        if miss_count >= 3:
                            is_lost = True
                else:
                    # Global rescue 256
                    r0 = cv2.resize(frame_buffer[0], (256, 144))
                    r1 = cv2.resize(frame_buffer[1], (256, 144))
                    r2 = cv2.resize(frame_buffer[2], (256, 144))
                    inp = np.concatenate([r0, r1, r2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    gx, gy = int(idx[-1]), int(idx[-2])
                    if conf >= thresh:
                        dx = float(gx) * 2.0
                        dy = float(gy) * 2.0
                        pred_pt = (dx * scale_w, dy * scale_h)
                        last_x, last_y = dx, dy
                        is_lost = False
                        miss_count = 0
            else:
                # Proposed: KF Pred, Velocity Gate <= 45, Adaptive 160, 1-Frame Forward Fill
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
                    
                    c0, _ = fast_crop_orig(frame_buffer[0], crop_cx, crop_cy, crop_r)
                    c1, _ = fast_crop_orig(frame_buffer[1], crop_cx, crop_cy, crop_r)
                    c2, _ = fast_crop_orig(frame_buffer[2], crop_cx, crop_cy, crop_r)
                    inp = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    lx, ly = int(idx[-1]), int(idx[-2])
                    
                    dx = crop_cx - float(crop_r) + float(lx) * scale_factor
                    dy = crop_cy - float(crop_r) + float(ly) * scale_factor
                    jump_dist = math.hypot(dx - last_x, dy - last_y)
                    pass_vgate = (jump_dist <= 45.0)
                    
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
                        # 1-Frame Forward Fill
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
                    inp = np.concatenate([r0, r1, r2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    gx, gy = int(idx[-1]), int(idx[-2])
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
                
    cap.release()
    
    # Calculate Metrics
    TP, FP, FN = 0, 0, 0
    errors = []
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
    mean_err = float(np.mean(errors)) if len(errors) > 0 else 0.0
    
    return {
        "TP": TP, "FP": FP, "FN": FN,
        "prec": prec, "rec": rec, "f1": f1,
        "mean_err": mean_err,
        "total_frames": total_frames,
        "gt_visible": sum(1 for g in gt_dict.values() if g["vis"] == 1)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Comprehensive Dataset-Wide Tracking Evaluation ===")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    model = MiniROIUNet().to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints_v4_student_augmented/best_augmented.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load validation split clips
    split_file = PROJECT_ROOT / "dataset/processed/tracknet_dataset_512x288/splits/val.txt"
    with open(split_file, "r") as f:
        val_clips = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(val_clips)} validation clips from: {split_file.name}")
    
    clips_to_eval = []
    for item in val_clips:
        parts = item.split("__")
        part_name, match_name, clip_id = parts[0], parts[1], parts[2]
        video_p = PROJECT_ROOT / f"dataset/raw/{part_name}/{match_name}/video/{clip_id}.mp4"
        csv_p = PROJECT_ROOT / f"dataset/raw/{part_name}/{match_name}/csv/{clip_id}_ball.csv"
        if video_p.exists() and csv_p.exists():
            clips_to_eval.append((f"{match_name}_{clip_id}", video_p, csv_p))
            
    print(f"Verified {len(clips_to_eval)} testable clip pairs.\n")
    
    results_baseline = []
    results_proposed = []
    
    print("--- Evaluating Baseline (Static 128, No KF, No Gate) ---")
    for name, v_p, c_p in tqdm(clips_to_eval, desc="Baseline"):
        res = evaluate_clip(model, v_p, c_p, device, mode="baseline", thresh=0.35)
        res["name"] = name
        results_baseline.append(res)
        
    print("\n--- Evaluating Proposed (KF-Pred, Adaptive 160, Jump Gate, 1F Fill) ---")
    for name, v_p, c_p in tqdm(clips_to_eval, desc="Proposed"):
        res = evaluate_clip(model, v_p, c_p, device, mode="proposed", thresh=0.35)
        res["name"] = name
        results_proposed.append(res)
        
    # Summarize Statistics
    def compute_summary(res_list):
        precs = [r["prec"] for r in res_list]
        recs = [r["rec"] for r in res_list]
        f1s = [r["f1"] for r in res_list]
        errs = [r["mean_err"] for r in res_list if r["mean_err"] > 0]
        
        tot_TP = sum(r["TP"] for r in res_list)
        tot_FP = sum(r["FP"] for r in res_list)
        tot_FN = sum(r["FN"] for r in res_list)
        agg_prec = (tot_TP / (tot_TP + tot_FP) * 100.0) if (tot_TP + tot_FP) > 0 else 0.0
        agg_rec = (tot_TP / (tot_TP + tot_FN) * 100.0) if (tot_TP + tot_FN) > 0 else 0.0
        agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) > 0 else 0.0
        
        return {
            "mean_prec": np.mean(precs), "median_prec": np.median(precs),
            "mean_rec": np.mean(recs), "median_rec": np.median(recs),
            "mean_f1": np.mean(f1s), "median_f1": np.median(f1s),
            "mean_err": np.mean(errs), "median_err": np.median(errs),
            "tot_TP": tot_TP, "tot_FP": tot_FP, "tot_FN": tot_FN,
            "agg_prec": agg_prec, "agg_rec": agg_rec, "agg_f1": agg_f1
        }
        
    base_sum = compute_summary(results_baseline)
    prop_sum = compute_summary(results_proposed)
    
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'BASELINE':<20} | {'PROPOSED SYSTEM':<20} | {'DIFF':<12}")
    print("="*80)
    print(f"{'Mean Recall':<25} | {base_sum['mean_rec']:>6.2f}%              | {prop_sum['mean_rec']:>6.2f}%              | {prop_sum['mean_rec'] - base_sum['mean_rec']:>+6.2f}%")
    print(f"{'Median Recall':<25} | {base_sum['median_rec']:>6.2f}%              | {prop_sum['median_rec']:>6.2f}%              | {prop_sum['median_rec'] - base_sum['median_rec']:>+6.2f}%")
    print(f"{'Mean Precision':<25} | {base_sum['mean_prec']:>6.2f}%              | {prop_sum['mean_prec']:>6.2f}%              | {prop_sum['mean_prec'] - base_sum['mean_prec']:>+6.2f}%")
    print(f"{'Median Precision':<25} | {base_sum['median_prec']:>6.2f}%              | {prop_sum['median_prec']:>6.2f}%              | {prop_sum['median_prec'] - base_sum['median_prec']:>+6.2f}%")
    print(f"{'Mean F1-Score':<25} | {base_sum['mean_f1']:>6.2f}               | {prop_sum['mean_f1']:>6.2f}               | {prop_sum['mean_f1'] - base_sum['mean_f1']:>+6.2f}")
    print(f"{'Median F1-Score':<25} | {base_sum['median_f1']:>6.2f}               | {prop_sum['median_f1']:>6.2f}               | {prop_sum['median_f1'] - base_sum['median_f1']:>+6.2f}")
    print(f"{'Aggregate Precision':<25} | {base_sum['agg_prec']:>6.2f}%              | {prop_sum['agg_prec']:>6.2f}%              | {prop_sum['agg_prec'] - base_sum['agg_prec']:>+6.2f}%")
    print(f"{'Aggregate Recall':<25} | {base_sum['agg_rec']:>6.2f}%              | {prop_sum['agg_rec']:>6.2f}%              | {prop_sum['agg_rec'] - base_sum['agg_rec']:>+6.2f}%")
    print(f"{'Aggregate F1-Score':<25} | {base_sum['agg_f1']:>6.2f}               | {prop_sum['agg_f1']:>6.2f}               | {prop_sum['agg_f1'] - base_sum['agg_f1']:>+6.2f}")
    print(f"{'Mean Error (px)':<25} | {base_sum['mean_err']:>6.2f} px            | {prop_sum['mean_err']:>6.2f} px            | {prop_sum['mean_err'] - base_sum['mean_err']:>+6.2f} px")
    print("="*80)
    
    # Save detailed CSV
    out_csv = PROJECT_ROOT / "report_materials/val_suite_comprehensive_evaluation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clip", "Frames", "GT_Vis", 
                         "Base_TP", "Base_FP", "Base_FN", "Base_Prec", "Base_Rec", "Base_F1",
                         "Prop_TP", "Prop_FP", "Prop_FN", "Prop_Prec", "Prop_Rec", "Prop_F1"])
        for b, p in zip(results_baseline, results_proposed):
            writer.writerow([
                b["name"], b["total_frames"], b["gt_visible"],
                b["TP"], b["FP"], b["FN"], f"{b['prec']:.2f}", f"{b['rec']:.2f}", f"{b['f1']:.2f}",
                p["TP"], p["FP"], p["FN"], f"{p['prec']:.2f}", f"{p['rec']:.2f}", f"{p['f1']:.2f}"
            ])
    print(f"\nDetailed per-clip CSV saved to: {out_csv}")

if __name__ == "__main__":
    main()
