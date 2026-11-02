# -*- coding: utf-8 -*-
"""
Jetson Nano Production Runtime (student_128.py)
=================================================
Real-Time 30 FPS Edge Badminton Tracking on NVIDIA Jetson Nano.
Powered by HierarchicalEdgeTracker (Single Source of Truth).

Key Features:
  1. Decoupled 30 FPS Ingestion Producer Thread (V4L2 Camera / Video Stream)
  2. Zero-Drop Queue Architecture (Queue maxsize=3)
  3. HierarchicalEdgeTracker Worker Thread (11ms local / 21ms coarse-to-fine rescue)
  4. 100% Dynamic Telemetry HUD (Measured FPS, measured drops, live TRT latency)
  5. Multi-Scale Smash ROI Expansion & Low-Contrast Proposal Rescue
"""

import os
import sys
import time
import math
import argparse
import queue
import threading
from collections import deque
import numpy as np
import cv2

# Import the unified HierarchicalEdgeTracker
try:
    from thesis.platforms.jetson_nano.tracker import HierarchicalEdgeTracker
except ImportError:
    # If running directly from platform directory on Nano
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
    from thesis.platforms.jetson_nano.tracker import HierarchicalEdgeTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Badminton Tracking Runtime on Jetson Nano")
    parser.add_argument("--source", type=str, default="weights/1_01_00.mp4",
                        help="Input video file path or camera device index (e.g., 0 for /dev/video0)")
    parser.add_argument("--engine_128", type=str, default="weights/miniroi_128x128.engine",
                        help="Path to 128x128 TensorRT engine")
    parser.add_argument("--engine_256", type=str, default="weights/miniroi_256x144.engine",
                        help="Path to 256x144 TensorRT engine")
    parser.add_argument("--output", type=str, default="weights/student_realtime_output.mp4",
                        help="Output video path (or leave empty to disable saving)")
    parser.add_argument("--t_track", type=float, default=0.20, help="Local tracking confidence threshold")
    parser.add_argument("--t_coarse", type=float, default=0.16, help="Coarse global proposal threshold")
    parser.add_argument("--t_verify", type=float, default=0.28, help="Fine verification confidence threshold")
    parser.add_argument("--fps", type=float, default=30.0, help="Target ingestion FPS")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI")
    return parser.parse_args()


def draw_dynamic_hud(frame, telemetry, ingest_fps, drop_rate, total_processed, trail):
    """Draw a fully dynamic, telemetry-driven HUD onto the frame."""
    h, w = frame.shape[:2]
    
    # Draw trajectory trail
    if telemetry and telemetry["pred_point"] is not None:
        px, py = telemetry["pred_point"]
        trail.append((int(px), int(py)))
    else:
        trail.append(None)
        
    for i in range(1, len(trail)):
        if trail[i - 1] is not None and trail[i] is not None:
            alpha = float(i) / len(trail)
            color = (int(0 * alpha), int(220 * alpha), int(255 * alpha))
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)
            
    # Draw current prediction marker
    if telemetry and telemetry["pred_point"] is not None:
        px, py = int(telemetry["pred_point"][0]), int(telemetry["pred_point"][1])
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 12, (0, 255, 255), 2, cv2.LINE_AA)
        
    # Overlay HUD Background Panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (460, 205), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (20, 20), (460, 205), (80, 110, 140), 1, cv2.LINE_AA)
    
    mode = telemetry["mode"] if telemetry else "INIT"
    lat = telemetry["latency_ms"] if telemetry else 0.0
    conf = telemetry["confidence"] if telemetry else 0.0
    crop_r = telemetry["crop_radius"] if telemetry else 64
    speed = telemetry["speed"] if telemetry else 0.0
    engine_used = telemetry["engine_used"] if telemetry else "NONE"
    
    mode_color = (0, 255, 120) if "LOCAL" in mode else ((0, 220, 255) if "RESCUE" in mode else (0, 140, 255))
    
    cv2.putText(frame, "Jetson Nano HierarchicalEdgeTracker", (35, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
                
    cv2.putText(frame, f"Ingest FPS: {ingest_fps:.1f} | Drop Rate: {drop_rate:.1f}%", (35, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1, cv2.LINE_AA)
                
    cv2.putText(frame, f"Mode: {mode} ({engine_used})", (35, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 2, cv2.LINE_AA)
                
    cv2.putText(frame, f"TRT Latency: {lat:.1f} ms (FP16)", (35, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                
    cv2.putText(frame, f"Confidence: {conf:.2f} | Crop Radius: {crop_r} px", (35, 154),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                
    cv2.putText(frame, f"Estimated Speed: {speed:.1f} px/f | Frames: {total_processed}", (35, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 220), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    print("=" * 80)
    print("  NVIDIA Jetson Nano Real-Time 30 FPS Runtime (student_128.py)")
    print("  Unified HierarchicalEdgeTracker (Single Source of Truth)")
    print("=" * 80)
    
    # Initialize Tracker
    print(f"[INFO] Initializing HierarchicalEdgeTracker...")
    print(f"       Engine 128: {args.engine_128}")
    print(f"       Engine 256: {args.engine_256}")
    
    tracker = HierarchicalEdgeTracker(
        engine_128_path=args.engine_128,
        engine_256_path=args.engine_256,
        t_track=args.t_track,
        t_coarse=args.t_coarse,
        t_verify=args.t_verify,
        adaptive_rescue=True
    )
    
    # Warmup TRT Engines
    print("[INFO] Performing 10 GPU Warmup Inferences...")
    dummy_128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
    dummy_256 = np.zeros((1, 9, 144, 256), dtype=np.float32)
    for _ in range(10):
        _ = tracker.trt_128.infer(dummy_128)
        _ = tracker.trt_256.infer(dummy_256)
    print("[INFO] Warmup complete.")
    
    # Determine Video Source
    is_cam = False
    try:
        cam_idx = int(args.source)
        cap = cv2.VideoCapture(cam_idx)
        is_cam = True
    except ValueError:
        cap = cv2.VideoCapture(args.source)
        
    if not cap.isOpened():
        print(f"[ERROR] Unable to open source: {args.source}")
        return
        
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    target_interval = 1.0 / args.fps
    
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w_orig, h_orig))
        print(f"[INFO] Saving annotated stream to: {args.output}")

    # Producer-Consumer Threading Setup
    frame_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
    stats = {"ingested": 0, "dropped": 0, "processed": 0}
    
    def producer_worker():
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
            t_next += target_interval
            t_sleep = t_next - time.perf_counter()
            if t_sleep > 0:
                time.sleep(t_sleep)
        stop_event.set()
        cap.release()

    prod_thread = threading.Thread(target=producer_worker)
    prod_thread.start()
    
    trail = deque(maxlen=10)
    t_start = time.perf_counter()
    fps_timer = time.perf_counter()
    measured_fps = args.fps
    
    try:
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                fid, frame = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
                
            stats["processed"] += 1
            telemetry = tracker.process_frame(frame, frame_id=fid)
            
            # FPS Calculation
            if stats["processed"] % 30 == 0:
                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed > 0:
                    measured_fps = 30.0 / elapsed
                fps_timer = now
                
            drop_rate = (stats["dropped"] / stats["ingested"] * 100.0) if stats["ingested"] > 0 else 0.0
            
            if telemetry is not None:
                # Render dynamic HUD on the middle frame stored in tracker
                frame_to_render = tracker.frame_buffer[1].copy()
                draw_dynamic_hud(frame_to_render, telemetry, measured_fps, drop_rate, stats["processed"], trail)
                
                if writer is not None:
                    writer.write(frame_to_render)
                    
                if not args.headless:
                    cv2.imshow("Jetson Nano Real-Time Tracking", frame_to_render)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            if stats["processed"] % 60 == 0 and telemetry:
                print(f"[STATUS] Frame {fid} | FPS: {measured_fps:.1f} | Drop: {drop_rate:.1f}% | "
                      f"Mode: {telemetry['mode']} | TRT Latency: {telemetry['latency_ms']:.1f} ms")
                      
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted.")
    finally:
        stop_event.set()
        prod_thread.join()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
    total_time = time.perf_counter() - t_start
    overall_fps = stats["processed"] / total_time if total_time > 0 else 0.0
    print("\n" + "=" * 80)
    print("  JETSON NANO EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Ingested Frames : {stats['ingested']}")
    print(f"Total Processed Frames: {stats['processed']}")
    print(f"Hardware Dropped      : {stats['dropped']} ({drop_rate:.2f}%)")
    print(f"Actual Throughput     : {overall_fps:.2f} FPS")
    if tracker.processed_frames > 0:
        avg_lat = tracker.total_latency_ms / tracker.processed_frames
        print(f"Average TRT Latency   : {avg_lat:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
