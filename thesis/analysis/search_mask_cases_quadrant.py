# -*- coding: utf-8 -*-
"""
Search and Generate 4-Quadrant Visual Comparison Cases for Motion Mask:
Case 1: Raw peak on noise/distractor -> After Mask, shifted to true shuttlecock!
Case 2: Raw peak on noise >= 0.55 threshold -> After Mask, noise suppressed below threshold!
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path
import csv
import math
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from thesis.models.miniroi_unet import MiniROIUNet

def get_peak(hmap):
    val = float(np.max(hmap))
    if val == 0:
        return 0, 0, 0.0
    idx = np.unravel_index(np.argmax(hmap), hmap.shape)
    return int(idx[-1]), int(idx[-2]), val

def create_quadrant_image(rgb_frame, mask, raw_hmap, masked_hmap,
                          title, sub3, sub4,
                          gt_pt=None, raw_pt=None, masked_pt=None):
    tile_w, tile_h = 512, 288
    rgb_vis = cv2.resize(rgb_frame, (tile_w, tile_h))
    
    # 1. Input RGB with Ground Truth
    panel1 = rgb_vis.copy()
    if gt_pt is not None:
        gx, gy = int(gt_pt[0] * tile_w / rgb_frame.shape[1]), int(gt_pt[1] * tile_h / rgb_frame.shape[0])
        cv2.circle(panel1, (gx, gy), 10, (0, 255, 0), 2)
        cv2.putText(panel1, "GT Ball", (gx + 12, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # 2. Binary Motion Mask
    mask_u8 = np.clip(cv2.resize(mask, (tile_w, tile_h)) * 255.0, 0, 255).astype(np.uint8)
    panel2 = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    # 3. Raw Heatmap Overlay
    raw_u8 = np.clip(cv2.resize(raw_hmap, (tile_w, tile_h)) * 255.0, 0, 255).astype(np.uint8)
    raw_color = cv2.applyColorMap(raw_u8, cv2.COLORMAP_JET)
    panel3 = cv2.addWeighted(rgb_vis, 0.35, raw_color, 0.65, 0)
    if raw_pt is not None:
        rx, ry, r_val = raw_pt
        prx, pry = int(rx * tile_w / raw_hmap.shape[1]), int(ry * tile_h / raw_hmap.shape[0])
        cv2.drawMarker(panel3, (prx, pry), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(panel3, f"False Peak: {r_val:.2f}", (max(10, prx - 60), max(20, pry - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    # 4. Masked Heatmap Overlay
    m_u8 = np.clip(cv2.resize(masked_hmap, (tile_w, tile_h)) * 255.0, 0, 255).astype(np.uint8)
    m_color = cv2.applyColorMap(m_u8, cv2.COLORMAP_JET)
    panel4 = cv2.addWeighted(rgb_vis, 0.35, m_color, 0.65, 0)
    if masked_pt is not None:
        mx, my, m_val = masked_pt
        pmx, pmy = int(mx * tile_w / masked_hmap.shape[1]), int(my * tile_h / masked_hmap.shape[0])
        cv2.drawMarker(panel4, (pmx, pmy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(panel4, f"Rescued True: {m_val:.2f}", (max(10, pmx - 65), max(20, pmy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    def add_header(img, text, bg_color=(25, 25, 25), text_color=(255, 255, 255)):
        hdr_h = 36
        c = np.zeros((img.shape[0] + hdr_h, img.shape[1], 3), dtype=np.uint8)
        c[:hdr_h, :] = bg_color
        c[hdr_h:, :] = img
        cv2.putText(c, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)
        return c

    t1 = add_header(panel1, "1. Input Video Frame (Middle Frame)")
    t2 = add_header(panel2, "2. Spatiotemporal Motion Mask (|It - It-1| >= 10)")
    t3 = add_header(panel3, sub3, bg_color=(15, 20, 45), text_color=(100, 200, 255))
    t4 = add_header(panel4, sub4, bg_color=(15, 40, 20), text_color=(120, 255, 120))

    grid = np.vstack([np.hstack([t1, t2]), np.hstack([t3, t4])])
    return grid

def scan_videos():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    weights_path = PROJECT_ROOT / "checkpoints_v4_student" / "epoch_30.pth"
    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    output_dir = PROJECT_ROOT / "report_materials" / "ablation_study" / "mask_golden_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case1_peak_rescued").mkdir(exist_ok=True)
    (output_dir / "case2_false_alarm_blocked").mkdir(exist_ok=True)

    # Videos to inspect
    test_videos = []
    
    # 1. Part1 Match9 & Match1
    for m in ["match9", "match1", "match3", "match12"]:
        v_dir = PROJECT_ROOT / "dataset" / "raw" / "part1" / m / "video"
        c_dir = PROJECT_ROOT / "dataset" / "raw" / "part1" / m / "csv"
        if v_dir.exists():
            for v_file in sorted(v_dir.glob("*.mp4")):
                c_file = c_dir / f"{v_file.stem}_ball.csv"
                if c_file.exists():
                    test_videos.append((v_file, c_file))

    # 2. am_singles
    am_base = PROJECT_ROOT / "dataset" / "raw" / "am_singles" / "am_singles"
    if am_base.exists():
        for m_dir in sorted(am_base.glob("match*")):
            for v_file in sorted((m_dir / "rally_video").glob("*.mp4")):
                c_file = m_dir / "ball_trajectory" / f"{v_file.stem}_ball.csv"
                if c_file.exists():
                    test_videos.append((v_file, c_file))

    print(f"Total video clips to scan: {len(test_videos)}")

    case1_found = []
    case2_found = []

    for v_idx, (v_path, c_path) in enumerate(test_videos):
        # Read GT
        gt_dict = {}
        with open(c_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                f_key = 'Frame' if 'Frame' in r else 'frame_num'
                v_key = 'Visibility' if 'Visibility' in r else 'visible'
                x_key = 'X' if 'X' in r else 'x'
                y_key = 'Y' if 'Y' in r else 'y'
                gt_dict[int(r[f_key])] = {
                    'vis': int(r[v_key]),
                    'x': float(r[x_key]),
                    'y': float(r[y_key])
                }

        cap = cv2.VideoCapture(str(v_path))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        scale_x = 512.0 / float(w_orig)
        scale_y = 288.0 / float(h_orig)

        frames = []
        fid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            f_train = cv2.resize(frame, (512, 288))
            frames.append(f_train)

            if len(frames) == 3:
                # Middle frame is target
                target_fid = fid - 1
                if target_fid in gt_dict:
                    gt = gt_dict[target_fid]
                    gt_x_512 = gt['x'] * scale_x
                    gt_y_512 = gt['y'] * scale_y

                    # Motion Mask between frame 0 and frame 1
                    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                    _, m = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                    m_float = m.astype(np.float32) / 255.0

                    # Model inference
                    x = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                    tx = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        raw_hmap = model(tx).squeeze().cpu().numpy()

                    masked_hmap = raw_hmap * cv2.resize(m_float, (512, 288))

                    rx, ry, r_val = get_peak(raw_hmap)
                    mx, my, m_val = get_peak(masked_hmap)

                    # Distance from raw peak to GT
                    dist_raw = math.hypot(rx - gt_x_512, ry - gt_y_512) if gt['vis'] == 1 else 999.0
                    dist_masked = math.hypot(mx - gt_x_512, my - gt_y_512) if gt['vis'] == 1 else 999.0
                    peak_shift = math.hypot(rx - mx, ry - my)

                    # Case 1: Peak Rescued
                    # Raw peak was false (dist_raw > 25), Masked peak is true (dist_masked <= 15), r_val >= 0.50, m_val >= 0.50
                    if gt['vis'] == 1 and dist_raw > 25.0 and dist_masked <= 15.0 and r_val >= 0.50 and m_val >= 0.50:
                        save_name = f"case1_rescued_{v_path.stem}_frame_{target_fid:05d}_raw{r_val:.2f}_masked{m_val:.2f}.jpg"
                        save_path = output_dir / "case1_peak_rescued" / save_name
                        grid = create_quadrant_image(
                            frames[1], m_float, raw_hmap, masked_hmap,
                            title=f"Case 1: Peak Rescued (Clip: {v_path.stem}, Frame {target_fid})",
                            sub3=f"3. BEFORE Mask: False Peak ({r_val:.2f}) on Noise (Dist: {dist_raw:.1f}px)",
                            sub4=f"4. AFTER Mask: Rescued True Peak ({m_val:.2f}) on Shuttlecock (Dist: {dist_masked:.1f}px)",
                            gt_pt=(gt['x'], gt['y']),
                            raw_pt=(rx, ry, r_val),
                            masked_pt=(mx, my, m_val)
                        )
                        cv2.imwrite(str(save_path), grid)
                        case1_found.append({
                            'video': v_path.name,
                            'frame': target_fid,
                            'raw_val': r_val,
                            'masked_val': m_val,
                            'dist_raw': dist_raw,
                            'dist_masked': dist_masked,
                            'path': str(save_path)
                        })
                        print(f"[FOUND CASE 1] {save_name} | Raw ({r_val:.2f}, dist {dist_raw:.1f}px) -> Masked ({m_val:.2f}, dist {dist_masked:.1f}px)")

                    # Case 2: False Alarm Blocked
                    # Raw peak was a false alarm >= 0.55 (either ball invisible or dist_raw > 30), and masked peak drops < 0.50
                    is_false_alarm = (gt['vis'] == 0) or (dist_raw > 30.0)
                    mask_val_at_raw = m_float[ry, rx] if ry < m_float.shape[0] and rx < m_float.shape[1] else 1.0

                    if is_false_alarm and r_val >= 0.55 and m_val < 0.45 and mask_val_at_raw == 0:
                        save_name = f"case2_blocked_{v_path.stem}_frame_{target_fid:05d}_raw{r_val:.2f}_masked{m_val:.2f}.jpg"
                        save_path = output_dir / "case2_false_alarm_blocked" / save_name
                        grid = create_quadrant_image(
                            frames[1], m_float, raw_hmap, masked_hmap,
                            title=f"Case 2: False Alarm Blocked (Clip: {v_path.stem}, Frame {target_fid})",
                            sub3=f"3. BEFORE Mask: False Alarm ({r_val:.2f} >= 0.55 Threshold)",
                            sub4=f"4. AFTER Mask: Suppressed ({m_val:.2f} < 0.55 Threshold -> False Alarm Blocked)",
                            gt_pt=(gt['x'], gt['y']) if gt['vis'] == 1 else None,
                            raw_pt=(rx, ry, r_val),
                            masked_pt=None
                        )
                        cv2.imwrite(str(save_path), grid)
                        case2_found.append({
                            'video': v_path.name,
                            'frame': target_fid,
                            'raw_val': r_val,
                            'masked_val': m_val,
                            'dist_raw': dist_raw,
                            'path': str(save_path)
                        })
                        print(f"[FOUND CASE 2] {save_name} | Raw ({r_val:.2f} >= 0.55) -> Masked ({m_val:.2f} < 0.55)")

                frames.pop(0)

            fid += 1

        cap.release()
        if (v_idx + 1) % 5 == 0 or len(case1_found) >= 3 and len(case2_found) >= 3:
            print(f"Scanned {v_idx+1}/{len(test_videos)} clips. Found Case 1: {len(case1_found)}, Case 2: {len(case2_found)}")
            if len(case1_found) >= 5 and len(case2_found) >= 5:
                break

    print("\n========================================================")
    print(f"SEARCH COMPLETE! Case 1: {len(case1_found)}, Case 2: {len(case2_found)}")
    print(f"Results saved in: {output_dir}")
    print("========================================================")

if __name__ == "__main__":
    scan_videos()
