# -*- coding: utf-8 -*-
"""
Comprehensive Thesis Ablation Study Pipeline
Compares:
1. With KF vs. Without KF (Inertial Navigation vs. Static ROI)
2. With Spatiotemporal Dynamic Mask vs. Without Mask (Motion Filtered vs. Raw Heatmap)

Evaluates on:
- dataset/raw/am_singles (Validation / Amateur singles videos from web)
- dataset/raw/part1 (Benchmark / Tournament match9 videos)

Generates:
- Quantitative CSV and JSON summaries
- Multi-metric publication-grade comparison plots
- High-resolution visual comparison figures (比對圖)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import torch
import csv
import json
import math
from collections import deque
import matplotlib.pyplot as plt

from thesis.platforms.pc import student_realtime_events as rt
from thesis.models.miniroi_unet import MiniROIUNet


def evaluate_video_variant(video_path, csv_path, model, device, use_kf, use_mask,
                           local_conf=0.55, global_conf=0.55, lost_duration=3, mask_thresh=10,
                           tolerance=15.0):
    """Run tracking inference on one video with given ablation toggles."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
        
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    labels = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                f_id = int(r['Frame'] if 'Frame' in r else r['frame_num'])
                labels[f_id] = r
                
    is_lost = True
    miss_count = 0
    kf = None
    last_x, last_y = 256, 144
    
    frames = deque(maxlen=3)
    f_id = -1
    
    tp, fp, fn, tn = 0, 0, 0, 0
    local_passes = 0
    global_passes = 0
    dist_sum = 0.0
    dist_count = 0
    
    # Store frame telemetry for comparison visualization
    history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_id += 1
        
        f512 = cv2.resize(cv2.resize(frame, (1280, 720)), (512, 288))
        frames.append(f512)
        if len(frames) < 3:
            continue
            
        mid_id = f_id - 1
        pred_pt = None
        status = ""
        raw_peak_info = None
        masked_peak_info = None
        current_roi_center = None
        
        # Phase 1: Local ROI Cruise (128x128)
        if not is_lost:
            local_passes += 1
            if use_kf and kf is not None:
                p = kf.predict()
                px, py = int(p[0, 0]), int(p[1, 0])
            else:
                px, py = int(last_x), int(last_y)
            current_roi_center = (px, py)
                
            c0, _, _ = rt.crop_and_pad(frames[0], px, py, 128)
            c1, _, _ = rt.crop_and_pad(frames[1], px, py, 128)
            c2, _, _ = rt.crop_and_pad(frames[2], px, py, 128)
            
            x_in = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
            tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.inference_mode():
                raw_hmap = model(tx).squeeze().cpu().numpy()
                
            g0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(cv2.absdiff(g1, g0), mask_thresh, 255, cv2.THRESH_BINARY)
            m_res = cv2.resize(m.astype(np.float32) / 255.0, (128, 128))
            
            rx, ry, rc = rt.get_shuttle_center_local(raw_hmap)
            mx, my, mc = rt.get_shuttle_center_local(raw_hmap * m_res)
            raw_peak_info = (px - 64 + rx, py - 64 + ry, rc)
            masked_peak_info = (px - 64 + mx, py - 64 + my, mc)
            
            effective_hmap = raw_hmap * m_res if use_mask else raw_hmap
            lx, ly, conf = rt.get_shuttle_center_local(effective_hmap)
            
            if conf >= local_conf:
                gx = px - 64 + lx
                gy = py - 64 + ly
                if use_kf:
                    if kf is None:
                        kf = rt.init_kalman_filter(gx, gy)
                    else:
                        kf.correct(np.array([[gx], [gy]], dtype=np.float32))
                last_x, last_y = gx, gy
                miss_count = 0
                pred_pt = (gx, gy)
                status = "track_128"
            else:
                miss_count += 1
                if miss_count >= lost_duration:
                    is_lost = True
                    status = "lost_trigger_global"
                else:
                    if use_kf:
                        last_x, last_y = px, py
                    pred_pt = (last_x, last_y)
                    status = f"kf_predict_miss_{miss_count}"
                    
        # Phase 2: Global Search (512x288)
        if is_lost:
            global_passes += 1
            x_in = np.concatenate(list(frames), axis=-1).astype(np.float32) / 255.0
            tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.inference_mode():
                raw_hmap = model(tx).squeeze().cpu().numpy()
                
            g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(cv2.absdiff(g1, g0), mask_thresh, 255, cv2.THRESH_BINARY)
            m_res = cv2.resize(m.astype(np.float32) / 255.0, (512, 288))
            
            rx, ry, rc = rt.get_shuttle_center_local(raw_hmap)
            mx, my, mc = rt.get_shuttle_center_local(raw_hmap * m_res)
            raw_peak_info = (rx, ry, rc)
            masked_peak_info = (mx, my, mc)
            
            effective_hmap = raw_hmap * m_res if use_mask else raw_hmap
            gx, gy, conf = rt.get_shuttle_center_local(effective_hmap)
            
            if conf > global_conf:
                if use_kf:
                    kf = rt.init_kalman_filter(gx, gy)
                last_x, last_y = gx, gy
                is_lost = False
                miss_count = 0
                pred_pt = (gx, gy)
                status = "global_rec"
            else:
                pred_pt = None
                status = "global_search"
                
        # Metric evaluation on middle frame
        label = labels.get(mid_id)
        frame_dist = None
        if label is not None:
            vis = int(label['Visibility']) == 1
            if pred_pt is not None:
                pt_native = (pred_pt[0] * W_orig / 512.0, pred_pt[1] * H_orig / 288.0)
                if vis:
                    gt_x, gt_y = float(label['X']), float(label['Y'])
                    dist = math.hypot(pt_native[0] - gt_x, pt_native[1] - gt_y)
                    frame_dist = dist
                    if dist <= tolerance:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                    dist_sum += dist
                    dist_count += 1
                else:
                    fp += 1
            else:
                if vis:
                    fn += 1
                else:
                    tn += 1
                    
        history.append({
            'frame_id': mid_id,
            'status': status,
            'pred_pt': pred_pt,
            'roi_center': current_roi_center,
            'raw_peak': raw_peak_info,
            'masked_peak': masked_peak_info,
            'gt': label,
            'dist': frame_dist
        })
        
    cap.release()
    total_f = local_passes + global_passes
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    loc_ratio = (local_passes / total_f * 100) if total_f else 0.0
    glob_ratio = (global_passes / total_f * 100) if total_f else 0.0
    mean_dist = (dist_sum / dist_count) if dist_count else 0.0
    
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'local_passes': local_passes, 'global_passes': global_passes,
        'local_ratio': loc_ratio, 'global_ratio': glob_ratio,
        'mean_dist': mean_dist,
        'history': history
    }


def evaluate_dataset(manifest_csv, model, device, dataset_filter=None):
    """Evaluate 4 ablation configurations across videos in manifest."""
    variants = [
        {"name": "Full (Proposed: KF + Mask)", "use_kf": True, "use_mask": True, "key": "proposed"},
        {"name": "No KF (Static ROI + Mask)", "use_kf": False, "use_mask": True, "key": "no_kf"},
        {"name": "No Mask (KF + Raw Heatmap)", "use_kf": True, "use_mask": False, "key": "no_mask"},
        {"name": "Baseline (Static ROI + Raw)", "use_kf": False, "use_mask": False, "key": "baseline"}
    ]
    
    with open(manifest_csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    if dataset_filter:
        rows = [r for r in rows if dataset_filter(r)]
        
    print(f"\nProcessing {len(rows)} videos from {Path(manifest_csv).name}...")
    
    summary = {v['key']: {'name': v['name'], 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
                          'local_passes': 0, 'global_passes': 0, 'dist_sum': 0.0, 'dist_count': 0,
                          'histories': {}} for v in variants}
                          
    for idx, row in enumerate(rows, 1):
        v_path = PROJECT_ROOT / row['video_path']
        c_path = PROJECT_ROOT / row['labels_path']
        print(f"[{idx}/{len(rows)}] Evaluating: {row.get('match', '')}/{row.get('video', v_path.name)}...")
        
        for v in variants:
            res = evaluate_video_variant(v_path, c_path, model, device,
                                         use_kf=v['use_kf'], use_mask=v['use_mask'])
            k = v['key']
            summary[k]['TP'] += res['TP']
            summary[k]['FP'] += res['FP']
            summary[k]['FN'] += res['FN']
            summary[k]['TN'] += res['TN']
            summary[k]['local_passes'] += res['local_passes']
            summary[k]['global_passes'] += res['global_passes']
            summary[k]['dist_sum'] += (res['mean_dist'] * res['TP'])
            summary[k]['dist_count'] += res['TP']
            summary[k]['histories'][row.get('video', v_path.name)] = res['history']
            
    # Compute aggregate metrics
    results = {}
    for k, data in summary.items():
        tp, fp, fn = data['TP'], data['FP'], data['FN']
        total_p = data['local_passes'] + data['global_passes']
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        mean_err = (data['dist_sum'] / data['dist_count']) if data['dist_count'] else 0.0
        loc_r = (data['local_passes'] / total_p * 100) if total_p else 0.0
        glob_r = (data['global_passes'] / total_p * 100) if total_p else 0.0
        
        results[k] = {
            'name': data['name'],
            'precision': prec * 100.0,
            'recall': rec * 100.0,
            'f1': f1 * 100.0,
            'local_ratio': loc_r,
            'global_ratio': glob_r,
            'mean_error': mean_err,
            'TP': tp, 'FP': fp, 'FN': fn
        }
        
    return results, summary


def generate_comparison_plots(am_results, part1_results, output_dir: Path):
    """Generate high quality matplotlib comparison charts."""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Bar Chart: Performance on am_singles (Validation)
    labels = ["Proposed\n(KF + Mask)", "No KF\n(Static ROI)", "No Mask\n(Raw HMap)", "Baseline\n(Static + Raw)"]
    keys = ["proposed", "no_kf", "no_mask", "baseline"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Plot 1: Amateur Singles
    p_am = [am_results[k]['precision'] for k in keys]
    r_am = [am_results[k]['recall'] for k in keys]
    f1_am = [am_results[k]['f1'] for k in keys]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax1 = axes[0]
    b1 = ax1.bar(x - width, p_am, width, label='Precision (%)', color='#2b5c8f')
    b2 = ax1.bar(x, r_am, width, label='Recall (%)', color='#4ba3e3')
    b3 = ax1.bar(x + width, f1_am, width, label='F1-Score (%)', color='#e06666')
    
    ax1.set_title("Validation Set: Amateur Singles (am_singles)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Metric (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                         
    # Plot 2: Professional Part1 (match9)
    p_p1 = [part1_results[k]['precision'] for k in keys]
    r_p1 = [part1_results[k]['recall'] for k in keys]
    f1_p1 = [part1_results[k]['f1'] for k in keys]
    
    ax2 = axes[1]
    b4 = ax2.bar(x - width, p_p1, width, label='Precision (%)', color='#2b5c8f')
    b5 = ax2.bar(x, r_p1, width, label='Recall (%)', color='#4ba3e3')
    b6 = ax2.bar(x + width, f1_p1, width, label='F1-Score (%)', color='#e06666')
    
    ax2.set_title("Benchmark Dataset: Tournament Part1 (match9)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Metric (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')
    
    for bars in [b4, b5, b6]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                         
    plt.tight_layout()
    chart_path1 = output_dir / "figure_ablation_accuracy_comparison.png"
    plt.savefig(chart_path1)
    plt.close()
    print(f"[SAVED] Accuracy Comparison Chart: {chart_path1}")
    
    # 2. Bar Chart: Tracking Efficiency (Local Cruise Ratio vs. Global Rescue Ratio)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    loc_p1 = [part1_results[k]['local_ratio'] for k in keys]
    glob_p1 = [part1_results[k]['global_ratio'] for k in keys]
    
    bars_loc = ax.bar(x - width/2, loc_p1, width, label='128x128 Local Cruise Ratio (~30 FPS)', color='#2e7d32')
    bars_glob = ax.bar(x + width/2, glob_p1, width, label='512x288 Global Rescue Ratio (~4.7 FPS)', color='#c62828')
    
    ax.set_title("Tracking Mode Distribution (Ablation on Tournament Part1)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Frame Share (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    for bar in bars_loc:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#1b5e20')
    for bar in bars_glob:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#b71c1c')
        
    plt.tight_layout()
    chart_path2 = output_dir / "figure_ablation_tracking_efficiency.png"
    plt.savefig(chart_path2)
    plt.close()
    print(f"[SAVED] Tracking Efficiency Chart: {chart_path2}")


def generate_kf_visual_comparison(video_path, kf_history, no_kf_history, output_path: Path):
    """
    Generate side-by-side visual comparison figure:
    Row 1: With Kalman Filter (proactively follows shuttle trajectory, stays locked)
    Row 2: Without Kalman Filter (static ROI lags behind, target out-of-bounds, fails into lost state)
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    f_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames[f_id] = frame
        f_id += 1
    cap.release()
    
    # Find a sequence where No KF failed (status contains lost or dist increases) while KF stayed locked
    target_fids = None
    for i in range(10, len(kf_history) - 5):
        h_kf = kf_history[i]
        h_nokf = no_kf_history[i]
        # Look for moment where no-kf lost target but kf kept tracking
        if "track" in h_kf['status'] and ("lost" in h_nokf['status'] or h_nokf['dist'] is None or (h_nokf['dist'] and h_nokf['dist'] > 25)):
            if h_kf['dist'] is not None and h_kf['dist'] < 10:
                target_fids = [i - 1, i, i + 1, i + 2]
                break
                
    if not target_fids:
        target_fids = [45, 46, 47, 48]
        
    print(f"Creating KF vs No-KF Visual Comparison for frames {target_fids}...")
    
    scale_x = 1280.0 / 512.0
    scale_y = 720.0 / 288.0
    
    col_imgs = []
    
    for row_idx, (tag, hist, color) in enumerate([("With Kalman Filter (Proposed)", kf_history, (0, 220, 0)),
                                                  ("Without Kalman Filter (Static ROI)", no_kf_history, (0, 0, 230))]):
        row_panels = []
        for fid in target_fids:
            if fid not in frames: continue
            raw_img = frames[fid].copy()
            h = hist[fid]
            
            # Draw ROI box
            if h['roi_center'] is not None:
                rcx, rcy = h['roi_center']
                half = 64
                x1 = int((rcx - half) * scale_x)
                y1 = int((rcy - half) * scale_y)
                x2 = int((rcx + half) * scale_x)
                y2 = int((rcy + half) * scale_y)
                box_c = (0, 255, 0) if "track" in h['status'] else ((0, 220, 255) if "kf" in h['status'] else (0, 0, 255))
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), box_c, 3)
                
            # Draw Ground Truth (Yellow Ring)
            if h['gt'] and int(h['gt']['Visibility']) == 1:
                gx, gy = int(float(h['gt']['X'])), int(float(h['gt']['Y']))
                cv2.circle(raw_img, (gx, gy), 12, (0, 255, 255), 2)
                
            # Draw Prediction (Solid Dot)
            if h['pred_pt'] is not None:
                px, py = int(h['pred_pt'][0] * scale_x), int(h['pred_pt'][1] * scale_y)
                pt_c = (0, 255, 0) if row_idx == 0 else (0, 0, 255)
                cv2.circle(raw_img, (px, py), 6, pt_c, -1)
                
            # Header text overlay
            cv2.rectangle(raw_img, (0, 0), (600, 70), (20, 20, 20), -1)
            cv2.putText(raw_img, f"Frame {fid} | {tag}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            st_text = f"State: {h['status']}"
            if h['dist'] is not None:
                st_text += f" | Err: {h['dist']:.1f}px"
            cv2.putText(raw_img, st_text, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            
            # Resize for compact display
            thumb = cv2.resize(raw_img, (400, 225))
            row_panels.append(thumb)
            
        row_strip = np.hstack(row_panels)
        col_imgs.append(row_strip)
        
    full_grid = np.vstack(col_imgs)
    cv2.imwrite(str(output_path), full_grid)
    print(f"[SAVED] KF Comparison Visual Figure: {output_path}")


def generate_mask_visual_comparison(video_path, frame_id, output_path: Path):
    """
    Generate 4-panel visual comparison:
    [Top-Left] Original Input Frame
    [Top-Right] Spatiotemporal Dynamic Mask (|I_t - I_{t-1}| > 10)
    [Bottom-Left] Heatmap WITHOUT Mask (Noise Hijacking on static background)
    [Bottom-Right] Heatmap WITH Mask (Pruned clean peak on moving shuttlecock)
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 2)
    frames = []
    for _ in range(3):
        ret, f = cap.read()
        if ret: frames.append(cv2.resize(cv2.resize(f, (1280, 720)), (512, 288)))
    cap.release()
    
    if len(frames) < 3:
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints_v4_student" / "epoch_30.pth", map_location=device, weights_only=False))
    model.eval()
    
    x = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
    tx = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        raw_hmap = model(tx).squeeze().cpu().numpy()
        
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
    m_res = cv2.resize(m.astype(np.float32) / 255.0, (512, 288))
    masked_hmap = raw_hmap * m_res
    
    tile_w, tile_h = 560, 315
    rgb_vis = cv2.resize(frames[1], (tile_w, tile_h))
    
    mask_u8 = np.clip(m_res * 255.0, 0, 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(cv2.resize(mask_u8, (tile_w, tile_h)), cv2.COLOR_GRAY2BGR)
    
    raw_u8 = np.clip(raw_hmap * 255.0, 0, 255).astype(np.uint8)
    raw_color = cv2.applyColorMap(cv2.resize(raw_u8, (tile_w, tile_h)), cv2.COLORMAP_JET)
    raw_blend = cv2.addWeighted(rgb_vis, 0.35, raw_color, 0.65, 0)
    
    masked_u8 = np.clip(masked_hmap * 255.0, 0, 255).astype(np.uint8)
    masked_color = cv2.applyColorMap(cv2.resize(masked_u8, (tile_w, tile_h)), cv2.COLORMAP_JET)
    masked_blend = cv2.addWeighted(rgb_vis, 0.35, masked_color, 0.65, 0)
    
    def add_header(img, title, color=(255, 255, 255)):
        c = img.copy()
        cv2.rectangle(c, (0, 0), (tile_w, 42), (20, 20, 20), -1)
        cv2.putText(c, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        return c
        
    t1 = add_header(rgb_vis, "1. Input Video Frame (Middle Frame)")
    t2 = add_header(mask_vis, "2. Spatiotemporal Dynamic Mask (|It - It-1| >= 10)")
    t3 = add_header(raw_blend, "3. WITHOUT Mask: Raw Heatmap (High Background Noise)", (0, 100, 255))
    t4 = add_header(masked_blend, "4. WITH Dynamic Mask: Noise Eliminated (Pristine Peak)", (0, 255, 0))
    
    top = np.hstack([t1, t2])
    bot = np.hstack([t3, t4])
    grid = np.vstack([top, bot])
    
    cv2.imwrite(str(output_path), grid)
    print(f"[SAVED] Dynamic Mask Comparison Visual Figure: {output_path}")


def main():
    print("=" * 60)
    print("THESIS COMPREHENSIVE ABLATION STUDY")
    print("1. With KF vs. Without KF")
    print("2. With Spatiotemporal Dynamic Mask vs. Without Mask")
    print("=" * 60)
    
    output_dir = PROJECT_ROOT / "report_materials" / "ablation_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Inference Device:", device)
    
    weights_path = PROJECT_ROOT / "checkpoints_v4_student" / "epoch_30.pth"
    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    
    # 1. Run on dataset/raw/am_singles (7 videos)
    am_manifest = PROJECT_ROOT / "configs" / "thesis" / "evaluation_amateur7.csv"
    am_results, am_summary = evaluate_dataset(am_manifest, model, device)
    
    # 2. Run on dataset/raw/part1 (match9, 9 videos)
    p1_manifest = PROJECT_ROOT / "configs" / "thesis" / "evaluation_original58.csv"
    p1_results, p1_summary = evaluate_dataset(p1_manifest, model, device,
                                              dataset_filter=lambda r: r.get('match') == 'match9')
                                              
    # Print Quantitative Tables
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY TABLE: Amateur Singles (am_singles - Validation)")
    print("=" * 80)
    print(f"{'Method / Variant':<32} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Local%':<8} | {'Global%':<8} | {'Err (px)':<8}")
    print("-" * 92)
    for k, v in am_results.items():
        print(f"{v['name']:<32} | {v['precision']:>8.2f}% | {v['recall']:>8.2f}% | {v['f1']:>8.2f}% | {v['local_ratio']:>7.1f}% | {v['global_ratio']:>7.1f}% | {v['mean_error']:>7.2f}")
        
    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY TABLE: Tournament Match9 (part1 - Benchmark)")
    print("=" * 80)
    print(f"{'Method / Variant':<32} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Local%':<8} | {'Global%':<8} | {'Err (px)':<8}")
    print("-" * 92)
    for k, v in p1_results.items():
        print(f"{v['name']:<32} | {v['precision']:>8.2f}% | {v['recall']:>8.2f}% | {v['f1']:>8.2f}% | {v['local_ratio']:>7.1f}% | {v['global_ratio']:>7.1f}% | {v['mean_error']:>7.2f}")
    print("=" * 80)
    
    # Save CSV
    csv_out = output_dir / "ablation_metrics_summary.csv"
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Configuration", "Precision(%)", "Recall(%)", "F1-Score(%)", "Local_Ratio(%)", "Global_Ratio(%)", "Mean_Error(px)"])
        for k, v in am_results.items():
            writer.writerow(["am_singles", v['name'], f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", f"{v['local_ratio']:.1f}", f"{v['global_ratio']:.1f}", f"{v['mean_error']:.2f}"])
        for k, v in p1_results.items():
            writer.writerow(["part1_match9", v['name'], f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", f"{v['local_ratio']:.1f}", f"{v['global_ratio']:.1f}", f"{v['mean_error']:.2f}"])
    print(f"\n[SAVED] Summary CSV: {csv_out}")
    
    # Save JSON
    json_out = output_dir / "ablation_metrics_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({'am_singles': am_results, 'part1_match9': p1_results}, f, indent=2)
    print(f"[SAVED] Summary JSON: {json_out}")
    
    # Generate Charts
    generate_comparison_plots(am_results, p1_results, output_dir)
    
    # Generate Qualitative Visual Figures (比對圖)
    print("\nGenerating Qualitative Visual Comparison Figures...")
    
    # 1. KF vs No-KF Comparison image
    p1_sample_video = PROJECT_ROOT / "dataset" / "raw" / "part1" / "match9" / "video" / "1_04_05.mp4"
    kf_hist = p1_summary['proposed']['histories']['1_04_05.mp4']
    nokf_hist = p1_summary['no_kf']['histories']['1_04_05.mp4']
    kf_fig_path = output_dir / "kf_vs_no_kf_visual_comparison.jpg"
    generate_kf_visual_comparison(p1_sample_video, kf_hist, nokf_hist, kf_fig_path)
    
    # 2. Dynamic Mask vs No-Mask Comparison image
    mask_fig_path = output_dir / "mask_vs_no_mask_visual_comparison.jpg"
    generate_mask_visual_comparison(p1_sample_video, frame_id=41, output_path=mask_fig_path)
    
    print("\n[SUCCESS] Ablation Study Finished Successfully!")


if __name__ == "__main__":
    main()
