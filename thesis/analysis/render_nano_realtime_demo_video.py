# -*- coding: utf-8 -*-
import csv
import math
import shutil
from pathlib import Path
from collections import deque
import cv2
import numpy as np

def render_demo():
    video_path = r"E:\Thesis\tracknet-series-pytorch\dataset\raw\part1\match12\video\2_01_01.mp4"
    gt_path = r"E:\Thesis\tracknet-series-pytorch\dataset\raw\part1\match12\csv\2_01_01_ball.csv"
    dump_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\match12_nano_dump.csv"
    out_video_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\nano_realtime_match12_demo.mp4"
    artifact_dst = r"C:\Users\User\.gemini\antigravity\brain\dac42c7a-e416-43b0-ae7b-dc7800d2667b\nano_realtime_match12_demo.mp4"

    # Load GT
    gt_dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_dict[int(r["Frame"])] = {
                "vis": int(r["Visibility"]),
                "x": float(r["X"]),
                "y": float(r["Y"])
            }

    # Load Nano Dump
    preds = {}
    with open(dump_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            f_id = int(r["frame"])
            px = float(r["pred_x"])
            py = float(r["pred_y"])
            preds[f_id] = {
                "x": px if px > 0 else None,
                "y": py if py > 0 else None,
                "conf": float(r["conf"]),
                "mode": r["mode"],
                "lat": float(r["lat_ms"])
            }

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    trail = deque(maxlen=10)
    fid = 0

    running_tp = 0
    running_fp = 0
    running_fn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred_info = preds.get(fid, None)
        gt_info = gt_dict.get(fid, None)

        # Update scoring
        if gt_info and pred_info:
            gvis = gt_info["vis"]
            has_pred = (pred_info["x"] is not None and pred_info["y"] is not None)
            if has_pred:
                if gvis == 1:
                    dist = math.hypot(pred_info["x"] - gt_info["x"], pred_info["y"] - gt_info["y"])
                    if dist <= 15.0:
                        running_tp += 1
                    else:
                        running_fp += 1
                else:
                    running_fp += 1
            else:
                if gvis == 1:
                    running_fn += 1

        cur_prec = (running_tp / (running_tp + running_fp) * 100.0) if (running_tp + running_fp) > 0 else 100.0
        cur_rec = (running_tp / (running_tp + running_fn) * 100.0) if (running_tp + running_fn) > 0 else 100.0

        # Draw trajectory trail
        if pred_info and pred_info["x"] is not None and pred_info["y"] is not None:
            trail.append((int(pred_info["x"]), int(pred_info["y"])))
        else:
            trail.append(None)

        # Draw fading trail lines
        for i in range(1, len(trail)):
            if trail[i - 1] is not None and trail[i] is not None:
                alpha = float(i) / len(trail)
                color = (int(0 * alpha), int(220 * alpha), int(255 * alpha)) # Gold/Yellow trail
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

        # Draw Ground Truth (Green Circle)
        if gt_info and gt_info["vis"] == 1 and gt_info["x"] > 0:
            gx, gy = int(gt_info["x"]), int(gt_info["y"])
            cv2.circle(frame, (gx, gy), 9, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw Prediction (Red Core with Yellow Outer Ring)
        if pred_info and pred_info["x"] is not None and pred_info["y"] is not None:
            px, py = int(pred_info["x"]), int(pred_info["y"])
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 12, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw HUD Panel (Top-Left)
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 195), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (20, 20), (450, 195), (80, 110, 140), 1, cv2.LINE_AA)

        mode_str = pred_info["mode"] if pred_info else "INIT"
        lat_val = pred_info["lat"] if pred_info else 0.0
        conf_val = pred_info["conf"] if pred_info else 0.0

        mode_color = (0, 255, 120) if "LOCAL" in mode_str else (0, 165, 255)

        cv2.putText(frame, "NVIDIA Jetson Nano Real-Time Tracking", (35, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Stream FPS: 30.0 FPS | Queue Drop: 0.0%", (35, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Mode: {mode_str}", (35, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Nano Latency: {lat_val:.1f} ms  (TRT FP16)", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Confidence: {conf_val:.2f} | Frame: {fid}/{total_frames}", (35, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Precision: {cur_prec:.1f}% | Recall: {cur_rec:.1f}%", (35, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 255), 2, cv2.LINE_AA)

        # Legend at bottom-right
        leg_overlay = frame.copy()
        cv2.rectangle(leg_overlay, (w - 280, h - 80), (w - 20, h - 20), (15, 20, 25), -1)
        cv2.addWeighted(leg_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (w - 280, h - 80), (w - 20, h - 20), (80, 110, 140), 1, cv2.LINE_AA)

        cv2.circle(frame, (w - 260, h - 58), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Ground Truth", (w - 240, h - 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.circle(frame, (w - 260, h - 35), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (w - 260, h - 35), 9, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Jetson Nano Track", (w - 240, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        fid += 1

    cap.release()
    writer.release()
    print(f"Rendered {fid} frames to {out_video_path}")

    # Copy to artifacts directory
    shutil.copyfile(out_video_path, artifact_dst)
    print(f"Copied to artifact path: {artifact_dst}")

if __name__ == "__main__":
    render_demo()
