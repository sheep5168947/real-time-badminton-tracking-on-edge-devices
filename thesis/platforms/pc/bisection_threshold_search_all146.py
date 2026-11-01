# -*- coding: utf-8 -*-
"""
Golden Section / Bisection Bracket Search for Optimal Confidence Threshold.
Evaluates across ALL 146 videos (match1 ~ match23) in the standard TrackNet benchmark.
Precision: 2 decimal places (step = 0.01).
Maximizes Aggregate F1-Score while logging:
- Mean & Median Recall
- Mean & Median Precision
- Mean & Median F1
- Aggregate Precision, Recall, F1
- Total TP, FP, FN across ~58,000 frames
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

def evaluate_clip(model, video_path, csv_path, device, thresh):
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
    
    return {"TP": TP, "FP": FP, "FN": FN, "prec": prec, "rec": rec, "f1": f1}

def evaluate_full_dataset(model, clips, device, thresh):
    thresh = round(thresh, 2)
    results = []
    for name, v_p, c_p in tqdm(clips, desc=f"Eval T={thresh:.2f}", leave=False):
        res = evaluate_clip(model, v_p, c_p, device, thresh)
        results.append(res)
        
    precs = [r["prec"] for r in results]
    recs = [r["rec"] for r in results]
    f1s = [r["f1"] for r in results]
    
    tot_TP = sum(r["TP"] for r in results)
    tot_FP = sum(r["FP"] for r in results)
    tot_FN = sum(r["FN"] for r in results)
    agg_prec = (tot_TP / (tot_TP + tot_FP) * 100.0) if (tot_TP + tot_FP) > 0 else 0.0
    agg_rec = (tot_TP / (tot_TP + tot_FN) * 100.0) if (tot_TP + tot_FN) > 0 else 0.0
    agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec)) if (agg_prec + agg_rec) > 0 else 0.0
    
    summary = {
        "thresh": thresh,
        "mean_rec": float(np.mean(recs)), "median_rec": float(np.median(recs)),
        "mean_prec": float(np.mean(precs)), "median_prec": float(np.median(precs)),
        "mean_f1": float(np.mean(f1s)), "median_f1": float(np.median(f1s)),
        "agg_rec": agg_rec, "agg_prec": agg_prec, "agg_f1": agg_f1,
        "tot_TP": tot_TP, "tot_FP": tot_FP, "tot_FN": tot_FN
    }
    return summary

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*85)
    print("      GOLDEN SECTION / BISECTION SEARCH FOR OPTIMAL CONFIDENCE THRESHOLD")
    print("           Evaluating Across ALL 146 Videos (match1 ~ match23)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("="*85)
    
    model = MiniROIUNet().to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints_v4_student_augmented/best_augmented.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Collect all 146 videos in match1~match23
    part1_dir = PROJECT_ROOT / "dataset/raw/part1"
    clips = []
    for match_p in sorted(part1_dir.glob("match*")):
        for vid_p in sorted((match_p / "video").glob("*.mp4")):
            csv_p = match_p / "csv" / f"{vid_p.stem}_ball.csv"
            if csv_p.exists():
                clips.append((f"{match_p.name}_{vid_p.stem}", vid_p, csv_p))
                
    print(f"Total verified video-csv pairs: {len(clips)} across {len(set(c[1].parent.parent.name for c in clips))} matches.")
    
    eval_cache = {}
    history = []
    
    def get_eval(t):
        t = round(t, 2)
        if t not in eval_cache:
            res = evaluate_full_dataset(model, clips, device, t)
            eval_cache[t] = res
            history.append(res)
            print(f"  [T={t:.2f}] Mean Rec={res['mean_rec']:.2f}% | Mean Prec={res['mean_prec']:.2f}% | Agg F1={res['agg_f1']:.2f} | Mean F1={res['mean_f1']:.2f} (TP={res['tot_TP']}, FP={res['tot_FP']})")
        return eval_cache[t]
        
    # Golden Section Search on [0.18, 0.48]
    a = 0.18
    b = 0.48
    phi = (math.sqrt(5) - 1) / 2.0  # ~0.6180339887
    
    print("\n--- Phase 1: Golden Section Bracket Search (Precision 0.01) ---")
    
    c = round(b - phi * (b - a), 2)
    d = round(a + phi * (b - a), 2)
    
    fc = get_eval(c)["agg_f1"]
    fd = get_eval(d)["agg_f1"]
    
    # Run iterations until interval is <= 0.02
    step = 1
    while (b - a) > 0.02 and step <= 8:
        print(f"\nIteration {step}: Bracket [{a:.2f}, {b:.2f}], testing c={c:.2f} (F1={fc:.2f}), d={d:.2f} (F1={fd:.2f})")
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = round(b - phi * (b - a), 2)
            fc = get_eval(c)["agg_f1"]
        else:
            a = c
            c = d
            fc = fd
            d = round(a + phi * (b - a), 2)
            fd = get_eval(d)["agg_f1"]
        step += 1
        
    best_t = round((a + b) / 2.0, 2)
    best_res = get_eval(best_t)
    
    # Also evaluate endpoints 0.20, 0.30, 0.40 if not already in cache for complete comparison table
    for anchor in [0.20, 0.25, 0.30, 0.35, 0.40]:
        get_eval(anchor)
        
    # Save search history to CSV
    out_csv = PROJECT_ROOT / "report_materials/threshold_bisection_146videos_progress.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold", "Mean_Rec", "Median_Rec", "Mean_Prec", "Median_Prec", 
                         "Mean_F1", "Median_F1", "Agg_Rec", "Agg_Prec", "Agg_F1", "Total_TP", "Total_FP", "Total_FN"])
        for h in sorted(history, key=lambda x: x["thresh"]):
            writer.writerow([
                f"{h['thresh']:.2f}", f"{h['mean_rec']:.2f}", f"{h['median_rec']:.2f}",
                f"{h['mean_prec']:.2f}", f"{h['median_prec']:.2f}", f"{h['mean_f1']:.2f}", f"{h['median_f1']:.2f}",
                f"{h['agg_rec']:.2f}", f"{h['agg_prec']:.2f}", f"{h['agg_f1']:.2f}",
                h["tot_TP"], h["tot_FP"], h["tot_FN"]
            ])
            
    print("\n" + "="*95)
    print("                     ALL 146 VIDEOS THRESHOLD SWEEP SUMMARY TABLE")
    print("="*95)
    print(f"{'Thresh':<8} | {'Mean Rec':<10} | {'Median Rec':<11} | {'Mean Prec':<10} | {'Median Prec':<12} | {'Agg F1':<8} | {'Mean F1':<8} | {'TP':<6} | {'FP':<6}")
    print("-"*95)
    for h in sorted(history, key=lambda x: x["thresh"]):
        star = " (BEST)" if h["thresh"] == best_t else ""
        print(f"{h['thresh']:<8.2f} | {h['mean_rec']:>6.2f}%    | {h['median_rec']:>6.2f}%     | {h['mean_prec']:>6.2f}%    | {h['median_prec']:>6.2f}%      | {h['agg_f1']:>6.2f}   | {h['mean_f1']:>6.2f}   | {h['tot_TP']:<6} | {h['tot_FP']:<6}{star}")
    print("="*95)
    print(f"\nOptimal Threshold Identified: T = {best_t:.2f} (Aggregate F1 = {best_res['agg_f1']:.2f}, Mean F1 = {best_res['mean_f1']:.2f})")
    print(f"Saved full progression log to: {out_csv}")

if __name__ == "__main__":
    main()
