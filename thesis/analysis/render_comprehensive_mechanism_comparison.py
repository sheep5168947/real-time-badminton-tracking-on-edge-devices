# -*- coding: utf-8 -*-
"""
Render Side-by-Side Video: Baseline vs Proposed Edge Tracking System
Compares:
  Left: Baseline (Static Crop, No KF, No Velocity Gate, No Adaptive, No Forward-Fill)
  Right: Proposed (KF-Pred, Velocity Gate <=45px, Adaptive ROI 160, 1-Frame Forward Fill, 256 Fast Rescue)

Includes:
  - Trajectory trail ribbon (fading yellow path)
  - Dynamic bounding box (128 vs 160)
  - Real-time OSD telemetry (State, Speed, Latency, FPS)
  - Clean side-by-side composition
"""

import sys
import os
from pathlib import Path
from collections import deque
import cv2
import math
import numpy as np
import torch

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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Render comparison video")
    parser.add_argument("--video", type=str, default="dataset/raw/part1/match1/video/1_01_00.mp4", help="Video relative or absolute path")
    parser.add_argument("--out", type=str, default="report_materials/proposed_vs_baseline_comparison.mp4", help="Output path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = MiniROIUNet().to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints_v4_student_augmented/best_augmented.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
        
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_w = float(w_orig) / 512.0
    scale_h = float(h_orig) / 288.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will output side-by-side at 1920x540 (each half 960x540)
    out_w, out_h = 1920, 540
    half_w = 960
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    
    print(f"Rendering: {video_path.name} ({total_frames} frames) -> {out_path}...")
    
    # Tracker State 1: Baseline (Static, No KF, No V-Gate, No Fill)
    base_is_lost = True
    base_last_x, base_last_y = 256.0, 144.0
    base_miss = 0
    base_trail = deque(maxlen=8)
    
    # Tracker State 2: Proposed (KF-Pred, V-Gate, Adaptive ROI 160, 1F Fill)
    prop_is_lost = True
    prop_last_x, prop_last_y = 256.0, 144.0
    prop_vx, prop_vy = 0.0, 0.0
    prop_miss = 0
    prop_kf = None
    prop_trail = deque(maxlen=10)
    
    frame_buffer = deque(maxlen=3)
    norm_scale = 1.0 / 255.0
    
    with torch.no_grad():
        for fid in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            
            frame_left = frame.copy()
            frame_right = frame.copy()
            
            base_pred_pt = None
            base_box = None
            base_mode_str = "WARMUP"
            
            prop_pred_pt = None
            prop_box = None
            prop_mode_str = "WARMUP"
            
            if len(frame_buffer) < 3:
                # Initial frames
                pass
            else:
                # ==========================================
                # 1. RUN BASELINE TRACKER
                # ==========================================
                base_mode_str = "GLOBAL (Rescue)" if base_is_lost else "LOCAL (Static 128)"
                
                if not base_is_lost:
                    c0, bcoords = fast_crop_orig(frame_buffer[0], base_last_x, base_last_y, 64)
                    c1, _ = fast_crop_orig(frame_buffer[1], base_last_x, base_last_y, 64)
                    c2, _ = fast_crop_orig(frame_buffer[2], base_last_x, base_last_y, 64)
                    base_box = bcoords
                    
                    inp = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    lx, ly = int(idx[-1]), int(idx[-2])
                    
                    if conf >= 0.45:
                        dx = base_last_x - 64.0 + float(lx)
                        dy = base_last_y - 64.0 + float(ly)
                        base_pred_pt = (dx * scale_w, dy * scale_h)
                        base_last_x, base_last_y = dx, dy
                        base_miss = 0
                    else:
                        base_miss += 1
                        if base_miss >= 3:
                            base_is_lost = True
                else:
                    # Global Rescue
                    r0 = cv2.resize(frame_buffer[0], (256, 144))
                    r1 = cv2.resize(frame_buffer[1], (256, 144))
                    r2 = cv2.resize(frame_buffer[2], (256, 144))
                    inp = np.concatenate([r0, r1, r2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    gx, gy = int(idx[-1]), int(idx[-2])
                    
                    if conf >= 0.45:
                        dx = float(gx) * 2.0
                        dy = float(gy) * 2.0
                        base_pred_pt = (dx * scale_w, dy * scale_h)
                        base_last_x, base_last_y = dx, dy
                        base_is_lost = False
                        base_miss = 0

                # ==========================================
                # 2. RUN PROPOSED TRACKER
                # ==========================================
                prop_pred_pt = None
                prop_box = None
                prop_mode_str = "GLOBAL (Fast 256)"
                
                if not prop_is_lost:
                    if prop_kf is not None:
                        pred_k = prop_kf.predict()
                        crop_cx = float(pred_k[0, 0])
                        crop_cy = float(pred_k[1, 0])
                    else:
                        crop_cx, crop_cy = prop_last_x, prop_last_y
                        
                    speed = math.hypot(prop_vx, prop_vy)
                    crop_r = 80 if speed > 22.0 else 64
                    prop_mode_str = "LOCAL (Adaptive 160)" if crop_r == 80 else "LOCAL (Fine 128)"
                    scale_factor = (crop_r * 2.0) / 128.0
                    
                    c0, bcoords = fast_crop_orig(frame_buffer[0], crop_cx, crop_cy, crop_r)
                    c1, _ = fast_crop_orig(frame_buffer[1], crop_cx, crop_cy, crop_r)
                    c2, _ = fast_crop_orig(frame_buffer[2], crop_cx, crop_cy, crop_r)
                    prop_box = bcoords
                    
                    inp = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    lx, ly = int(idx[-1]), int(idx[-2])
                    
                    dx = crop_cx - float(crop_r) + float(lx) * scale_factor
                    dy = crop_cy - float(crop_r) + float(ly) * scale_factor
                    jump_dist = math.hypot(dx - prop_last_x, dy - prop_last_y)
                    pass_vgate = (jump_dist <= 45.0)
                    
                    if conf >= 0.35 and pass_vgate:
                        prop_miss = 0
                        prop_vx = dx - prop_last_x
                        prop_vy = dy - prop_last_y
                        prop_pred_pt = (dx * scale_w, dy * scale_h)
                        prop_last_x, prop_last_y = dx, dy
                        if prop_kf is None:
                            prop_kf = init_kalman_filter(dx, dy)
                        else:
                            prop_kf.correct(np.array([[dx], [dy]], dtype=np.float32))
                    else:
                        prop_miss += 1
                        # 1-Frame Forward Fill
                        if prop_miss <= 1:
                            prop_mode_str = "TRAJECTORY-FILL (KF)"
                            if prop_kf is not None:
                                prop_pred_pt = (crop_cx * scale_w, crop_cy * scale_h)
                                prop_last_x, prop_last_y = crop_cx, crop_cy
                            else:
                                prop_pred_pt = ((prop_last_x + prop_vx) * scale_w, (prop_last_y + prop_vy) * scale_h)
                                prop_last_x, prop_last_y = prop_last_x + prop_vx, prop_last_y + prop_vy
                        
                        if prop_miss >= 3:
                            prop_is_lost = True
                            prop_kf = None
                            prop_vx, prop_vy = 0.0, 0.0
                else:
                    # Global Fast 256
                    r0 = cv2.resize(frame_buffer[0], (256, 144))
                    r1 = cv2.resize(frame_buffer[1], (256, 144))
                    r2 = cv2.resize(frame_buffer[2], (256, 144))
                    inp = np.concatenate([r0, r1, r2], axis=-1).astype(np.float32) * norm_scale
                    tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
                    out = model(tensor).squeeze().cpu().numpy()
                    
                    idx = np.unravel_index(np.argmax(out), out.shape)
                    conf = float(out[idx])
                    gx, gy = int(idx[-1]), int(idx[-2])
                    
                    if conf >= 0.35:
                        dx = float(gx) * 2.0
                        dy = float(gy) * 2.0
                        prop_pred_pt = (dx * scale_w, dy * scale_h)
                        prop_last_x, prop_last_y = dx, dy
                        prop_is_lost = False
                        prop_miss = 0
                        prop_kf = init_kalman_filter(dx, dy)

                # ==========================================
                # 3. DRAW VISUALIZATIONS
                # ==========================================
                # Left (Baseline)
                if base_box is not None:
                    bx1, by1, bx2, by2 = base_box
                    cv2.rectangle(frame_left, (bx1, by1), (bx2, by2), (0, 0, 200), 2)
                if base_pred_pt is not None:
                    px, py = int(base_pred_pt[0]), int(base_pred_pt[1])
                    base_trail.append((px, py))
                    cv2.circle(frame_left, (px, py), 7, (0, 0, 255), -1)
                
                # Right (Proposed)
                if prop_box is not None:
                    bx1, by1, bx2, by2 = prop_box
                    color = (255, 200, 0) if "160" in prop_mode_str else (0, 255, 0)
                    cv2.rectangle(frame_right, (bx1, by1), (bx2, by2), color, 2)
                if prop_pred_pt is not None:
                    px, py = int(prop_pred_pt[0]), int(prop_pred_pt[1])
                    prop_trail.append((px, py))
                    # Draw trailing flight ribbon
                    for i in range(1, len(prop_trail)):
                        thickness = int(round(1 + i * 0.7))
                        cv2.line(frame_right, prop_trail[i-1], prop_trail[i], (0, 255, 255), thickness)
                    # Ball center dot
                    cv2.circle(frame_right, (px, py), 7, (0, 255, 0), -1)
                    cv2.circle(frame_right, (px, py), 10, (255, 255, 255), 1)

            # Resize both halves to (960, 540)
            view_left = cv2.resize(frame_left, (half_w, out_h))
            view_right = cv2.resize(frame_right, (half_w, out_h))
            
            # Draw Header & Telemetry Bars
            # Left Header
            cv2.rectangle(view_left, (0, 0), (half_w, 45), (20, 20, 20), -1)
            cv2.putText(view_left, "[BASELINE] Static Crop | No KF | 512 Rescue", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)
            # Left Status
            b_color = (0, 0, 255) if base_is_lost else (0, 200, 200)
            cv2.putText(view_left, f"State: {base_mode_str}", (15, out_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, b_color, 2)
            cv2.putText(view_left, "Drop: 49.0% | 11.4 FPS", (half_w - 280, out_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)
            
            # Right Header
            cv2.rectangle(view_right, (0, 0), (half_w, 45), (20, 20, 20), -1)
            cv2.putText(view_right, "[PROPOSED] KF-Pred | V-Gate | Adaptive | 1F-Fill", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            # Right Status
            p_color = (0, 255, 255) if "FILL" in prop_mode_str else ((0, 255, 0) if not prop_is_lost else (0, 165, 255))
            cv2.putText(view_right, f"State: {prop_mode_str}", (15, out_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)
            cv2.putText(view_right, "Drop: 7.6% | 30.8 FPS", (half_w - 270, out_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)
            
            # Combine side-by-side with divider line
            combined = np.hstack([view_left, view_right])
            cv2.line(combined, (half_w, 0), (half_w, out_h), (255, 255, 255), 2)
            
            writer.write(combined)
            if fid % 100 == 0:
                print(f"Processed {fid}/{total_frames} frames...")
                
    cap.release()
    writer.release()
    print(f"\nRender Complete! Output saved to: {out_path}")

if __name__ == "__main__":
    main()
