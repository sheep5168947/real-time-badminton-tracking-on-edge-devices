# -*- coding: utf-8 -*-
"""
Run Proposed HierarchicalEdgeTracker on Jetson Nano under strict 30 FPS real-time ingestion,
record frame predictions, and render an annotated high-resolution MP4 video.
"""
import os
import sys
import time
import math
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from thesis.platforms.jetson_nano.tracker import HierarchicalEdgeTracker

def main():
    video_path = "dataset/raw/test_video.mp4"
    out_video_path = "report_materials/test_video_nano_realtime_proposed.mp4"
    os.makedirs("report_materials", exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        sys.exit(1)
        
    engine_128 = "weights/miniroi_128x128_augmented.engine"
    engine_256 = "weights/miniroi_256x144_augmented.engine"
    
    print("Initializing HierarchicalEdgeTracker on Jetson Nano...")
    tracker = HierarchicalEdgeTracker(
        engine_128_path=engine_128,
        engine_256_path=engine_256,
        t_track=0.20,
        t_coarse=0.14,
        t_verify=0.35,
        max_miss=2,
        jump_gate=50.0,
        require_ver=True,
        confirm_rescue=True
    )
    
    # Warm up engines
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        tracker.process_frame(dummy, 0)
    tracker.reset()
    print("Tracker warmed up successfully!")
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video: {width}x{height}, {total_frames} frames @ {fps} FPS")
    
    # Dual-thread Real-Time Simulation
    frame_queue = Queue(maxsize=3)
    stop_event = threading.Event()
    dropped_frames = 0
    predictions = {}
    telemetry_log = []
    
    def producer():
        nonlocal dropped_frames
        cap_in = cv2.VideoCapture(video_path)
        fid = 0
        interval = 1.0 / 30.0
        
        while not stop_event.is_set():
            t_start = time.time()
            ret, frame = cap_in.read()
            if not ret:
                break
            
            item = (fid, frame)
            if frame_queue.full():
                dropped_frames += 1
                try:
                    _ = frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put(item)
            fid += 1
            
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        cap_in.release()
        frame_queue.put(None) # Sentinel
        print(f"Producer finished. Ingested frames: {fid}, Dropped: {dropped_frames}")
        
    p_thread = threading.Thread(target=producer, daemon=True)
    
    print("\n--- Starting Real-Time Ingestion Tracking on Jetson Nano ---")
    t_run_start = time.time()
    p_thread.start()
    
    processed_count = 0
    latencies = []
    
    while True:
        try:
            item = frame_queue.get(timeout=3.0)
        except Empty:
            break
            
        if item is None:
            break
            
        cur_fid, cur_frame = item
        t0 = time.time()
        telemetry = tracker.process_frame(cur_frame, frame_id=cur_fid)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)
        
        if telemetry is not None:
            mid_fid = telemetry["frame_id"]
            if telemetry.get("pred_point") is not None:
                predictions[mid_fid] = telemetry["pred_point"]
            if telemetry.get("retro_pred") is not None:
                r_fid, r_pt = telemetry["retro_pred"]
                predictions[r_fid] = r_pt
                
            telemetry_log.append({
                "fid": cur_fid,
                "mid_fid": mid_fid,
                "pred": predictions.get(mid_fid),
                "mode": telemetry["mode"],
                "conf": telemetry["confidence"],
                "lat_ms": dt
            })
            mode_display = telemetry["mode"]
        else:
            mode_display = "BUFFER_WARMUP"
        
        processed_count += 1
        if processed_count % 50 == 0 or processed_count == total_frames:
            avg_lat = np.mean(latencies[-50:])
            print(f"Frame {processed_count}/{total_frames} | Mode: {mode_display:16s} | Latency: {avg_lat:.1f}ms | Drops: {dropped_frames}")
            
    p_thread.join()
    t_total_time = time.time() - t_run_start
    effective_fps = processed_count / t_total_time if t_total_time > 0 else 0
    print(f"\nTracking Complete in {t_total_time:.2f}s | Effective FPS: {effective_fps:.2f} | Drops: {dropped_frames}")
    
    # Second Pass: High-Quality Video Rendering
    print(f"\nRendering annotated output video to {out_video_path}...")
    cap_render = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, 30.0, (width, height))
    
    trail = deque(maxlen=8)
    r_fid = 0
    
    while True:
        ret, frame = cap_render.read()
        if not ret:
            break
            
        # Draw Trailing Path
        pt = predictions.get(r_fid)
        if pt is not None:
            trail.append((int(pt[0]), int(pt[1])))
        else:
            if len(trail) > 0:
                trail.popleft()
                
        # Render Trail
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = int(2 + 4 * alpha)
            # Glowing cyan to red trail
            color = (int(0 + 255 * alpha), int(200 * (1 - alpha)), int(255 * (1 - alpha)))
            cv2.line(frame, trail[i-1], trail[i], color, thickness)
            
        # Render Current Prediction (Glowing Ring)
        if pt is not None:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(frame, (cx, cy), 16, (0, 0, 255), 3) # Outer red ring
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)  # Middle yellow ring
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)# White center dot
            
        # Overlay Modern HUD Top Bar
        # Semi-transparent dark banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # HUD Text
        cv2.putText(frame, "NVIDIA Jetson Nano 4GB (10W MAXN)", (25, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (56, 189, 248), 2, cv2.LINE_AA)
        
        status_mode = "TRACKING" if pt is not None else "SEARCHING"
        mode_col = (52, 211, 153) if pt is not None else (251, 191, 36)
        cv2.putText(frame, f"STATUS: [{status_mode}]", (25, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, mode_col, 2, cv2.LINE_AA)
                    
        # Metrics on right side
        avg_lat = np.mean(latencies) if len(latencies) > 0 else 12.0
        cv2.putText(frame, f"Real-Time: 30.0 FPS | Latency: {avg_lat:.1f}ms | Drops: {dropped_frames}",
                    (width - 680, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (248, 250, 252), 2, cv2.LINE_AA)
        
        writer.write(frame)
        r_fid += 1
        
    cap_render.release()
    writer.release()
    print(f"Video rendered successfully: {out_video_path} ({r_fid} frames)")

if __name__ == "__main__":
    main()
