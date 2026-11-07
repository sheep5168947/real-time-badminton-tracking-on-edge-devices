# -*- coding: utf-8 -*-
"""
HierarchicalEdgeTracker: Unified Edge Tracking Engine for NVIDIA Jetson Nano
=============================================================================
Single Source of Truth for:
  - Jetson Nano Production Runtime (student_128.py)
  - Real-Time Hardware Benchmark Suite (benchmark_nano_*.py)
  - Live Telemetry & Demonstration Renderer (render_*.py)

Core Capabilities:
  1. Two-Stage Coarse-to-Fine Pipeline:
     - Coarse Global Proposal (256x144 TRT Engine)
     - Fine Verification & Local Tracking (128x128 TRT Engine)
  2. Zero-Copy Incremental Ring Buffer (1-frame resize per ingestion)
  3. High-Velocity Smash Dynamic ROI Expansion (128 -> 160 px when speed > 20 px/frame)
  4. Adaptive Low-Contrast Proposal Rescue (Dynamic PMR Gate)
  5. Kinematic Pre-Serve & Stationary Ball Management
  6. Middle-Frame Temporal Alignment (f_{t-1} prediction from [f_{t-2}, f_{t-1}, f_t])
  7. Pure Dynamic Telemetry (Zero hardcoded metrics)
"""

import math
import time
from collections import deque
import numpy as np
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


class FastTRTModel:
    """Zero-overhead TensorRT async inference engine with page-locked host memory."""
    def __init__(self, engine_path, ctx=None):
        if not HAS_TRT:
            raise RuntimeError("TensorRT or PyCUDA is not installed on this environment.")
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'host': host_mem, 'device': device_mem,
                    'shape': self.engine.get_binding_shape(binding)
                })
            else:
                self.outputs.append({
                    'host': host_mem, 'device': device_mem,
                    'shape': self.engine.get_binding_shape(binding)
                })

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])


def fast_argmax(heatmap_2d):
    """Fast argmax extraction on 2D probability map."""
    idx = np.unravel_index(np.argmax(heatmap_2d), heatmap_2d.shape)
    conf = float(heatmap_2d[idx])
    return int(idx[-1]), int(idx[-2]), conf


def subpixel_centroid(heatmap_2d, lx, ly, win=2):
    """
    Sub-pixel centroid refinement using localized center-of-mass around argmax peak.
    Eliminates integer quantization jitter and physical spatial displacement noise.
    """
    h, w = heatmap_2d.shape
    x_min = max(0, lx - win)
    x_max = min(w, lx + win + 1)
    y_min = max(0, ly - win)
    y_max = min(h, ly + win + 1)
    patch = heatmap_2d[y_min:y_max, x_min:x_max]
    base = float(np.min(patch))
    weights = np.maximum(0.0, patch - base)
    total_w = float(np.sum(weights))
    if total_w < 1e-6:
        return float(lx), float(ly)
    ys, xs = np.indices(patch.shape)
    cx = float(np.sum((xs + x_min) * weights)) / total_w
    cy = float(np.sum((ys + y_min) * weights)) / total_w
    return cx, cy


def robust_crop(frame_orig, cx_512, cy_288, radius_512=64, target_size=128):
    """
    Sub-pixel-safe multi-scale ROI cropper.
    Crops centered at (cx_512, cy_288) with radius radius_512 in 512x288 coordinate space,
    handling image boundaries with zero-padding, then resizes to (target_size, target_size).
    """
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
        return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR), (x_min, y_min, x_max, y_max)
        
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
        
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR), (x_min, y_min, x_max, y_max)


def init_kalman(x, y):
    """
    Constant-velocity 2D kinematic Kalman Filter.
    Tuned for agile badminton tracking (responsive to true visual detections
    while rejecting high-frequency pixel jitter).
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
    return kf


class HierarchicalEdgeTracker:
    """
    Unified Hierarchical Edge Tracker.
    Coordinates coarse proposals, fine verification, Kalman kinematics,
    and adaptive multi-scale rescue.
    """
    def __init__(self,
                 engine_128_path="weights/miniroi_128x128_augmented.engine",
                 engine_256_path="weights/miniroi_256x144_augmented.engine",
                 t_track=0.20,
                 t_coarse=0.14,
                 t_verify=0.35,
                 max_miss=2,
                 jump_gate=50.0,
                 smash_speed_thresh=20.0,
                 adaptive_rescue=True,
                 vel_comp=0.28,
                 require_ver=True,
                 confirm_rescue=True):
        
        self.t_track = float(t_track)
        self.t_coarse = float(t_coarse)
        self.t_verify = float(t_verify)
        self.max_miss = int(max_miss)
        self.jump_gate = float(jump_gate)
        self.smash_speed_thresh = float(smash_speed_thresh)
        self.adaptive_rescue = bool(adaptive_rescue)
        self.vel_comp = float(vel_comp)
        self.require_ver = bool(require_ver)
        self.confirm_rescue = bool(confirm_rescue)
        
        # Load TensorRT Models (with graceful fallback to baseline if augmented missing)
        import os
        if not os.path.exists(engine_128_path) and os.path.exists("weights/miniroi_128x128.engine"):
            engine_128_path = "weights/miniroi_128x128.engine"
        if not os.path.exists(engine_256_path) and os.path.exists("weights/miniroi_256x144.engine"):
            engine_256_path = "weights/miniroi_256x144.engine"
            
        self.trt_128 = FastTRTModel(engine_128_path)
        self.trt_256 = FastTRTModel(engine_256_path)
        
        # Pinned Input Buffers
        self.buf_128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
        self.buf_256 = np.zeros((1, 9, 144, 256), dtype=np.float32)
        self.norm_scale = np.float32(1.0 / 255.0)
        
        # Circular Incremental Pre-resize Ring Buffer
        self.resized_slots = [np.zeros((3, 144, 256), dtype=np.float32) for _ in range(3)]
        self.slot_idx = 0
        self.num_cached = 0
        
        # Frame History Buffers
        self.frame_buffer = deque(maxlen=3)
        self.fid_buffer = deque(maxlen=3)
        
        # Tracking Kinematic State
        self.is_lost = True
        self.tentative = None
        self.kf = None
        self.last_x = 256.0
        self.last_y = 144.0
        self.vx = 0.0
        self.vy = 0.0
        self.miss_count = 0
        
        # Stationary / Pre-Serve Kinematic Manager
        self.stationary_frames = 0
        self.is_stationary = False
        
        # Cumulative Telemetry
        self.processed_frames = 0
        self.total_latency_ms = 0.0

    def reset(self):
        """Reset internal tracking state and history."""
        self.is_lost = True
        self.tentative = None
        self.kf = None
        self.last_x = 256.0
        self.last_y = 144.0
        self.vx = 0.0
        self.vy = 0.0
        self.miss_count = 0
        self.stationary_frames = 0
        self.is_stationary = False
        self.slot_idx = 0
        self.num_cached = 0
        self.frame_buffer.clear()
        self.fid_buffer.clear()

    def process_frame(self, frame_orig, frame_id=None):
        """
        Ingest a single frame, update the circular ring buffer, and execute
        the hierarchical edge tracking pipeline for the middle frame (f_{t-1}).

        Returns:
            telemetry: dict or None (if buffer is warming up < 3 frames)
        """
        if frame_id is None:
            frame_id = self.processed_frames
            
        h_f, w_f = frame_orig.shape[:2]
        scale_w = float(w_f) / 512.0
        scale_h = float(h_f) / 288.0
        
        self.frame_buffer.append(frame_orig)
        self.fid_buffer.append(frame_id)
        
        # Incremental single-frame pre-resize (amortized zero bottleneck)
        r_new = cv2.resize(frame_orig, (256, 144), interpolation=cv2.INTER_LINEAR)
        cur_slot = self.resized_slots[self.slot_idx % 3]
        cur_slot[:] = np.transpose(r_new, (2, 0, 1)) * self.norm_scale
        self.slot_idx += 1
        self.num_cached += 1
        
        if self.num_cached < 3:
            return None
            
        target_mid_fid = self.fid_buffer[1]
        t0 = time.perf_counter()
        
        pred_pt_orig = None
        pred_pt_512 = None
        retro_pred = None
        conf_out = 0.0
        mode_str = "SEARCH_LOST"
        engine_str = "NONE"
        crop_r = 64
        
        # =========================================================================
        # Stage 1: Local ROI Tracking Mode
        # =========================================================================
        if not self.is_lost:
            speed = math.hypot(self.vx, self.vy)
            if speed <= 20.0 and self.miss_count == 0:
                crop_cx = self.last_x + self.vx
                crop_cy = self.last_y + self.vy
            else:
                crop_cx = self.last_x
                crop_cy = self.last_y
                
            # Mechanism A: High-Velocity Smash Dynamic ROI Expansion
            # When speed > smash_speed_thresh, expand crop radius from 64 (128x128) to 80 (160x160)
            if speed > self.smash_speed_thresh:
                crop_r = 80
                gate = max(self.jump_gate * 1.5, speed * 1.6)
            else:
                crop_r = 64
                gate = self.jump_gate
                
            scale_factor = (crop_r * 2.0) / 128.0
            
            c0, _ = robust_crop(self.frame_buffer[0], crop_cx, crop_cy, crop_r, 128)
            c1, _ = robust_crop(self.frame_buffer[1], crop_cx, crop_cy, crop_r, 128)
            c2, _ = robust_crop(self.frame_buffer[2], crop_cx, crop_cy, crop_r, 128)
            
            self.buf_128[0, 0:3] = np.transpose(c0, (2, 0, 1)) * self.norm_scale
            self.buf_128[0, 3:6] = np.transpose(c1, (2, 0, 1)) * self.norm_scale
            self.buf_128[0, 6:9] = np.transpose(c2, (2, 0, 1)) * self.norm_scale
            
            out_128 = self.trt_128.infer(self.buf_128).squeeze()
            lx, ly, conf = fast_argmax(out_128)
            slx, sly = subpixel_centroid(out_128, lx, ly, win=2)
            engine_str = "128x128"
            
            dx = crop_cx - float(crop_r) + float(slx) * scale_factor
            dy = crop_cy - float(crop_r) + float(sly) * scale_factor
            jump = math.hypot(dx - self.last_x, dy - self.last_y)
            pass_gate = (jump <= gate)
            
            if conf >= self.t_track and pass_gate:
                self.miss_count = 0
                new_vx = dx - self.last_x
                new_vy = dy - self.last_y
                
                # Mechanism C: Kinematic Pre-Serve & Stationary Detection
                disp = math.hypot(new_vx, new_vy)
                if disp < 1.5:
                    self.stationary_frames += 1
                else:
                    self.stationary_frames = 0
                    self.is_stationary = False
                    
                # Terminate tracking if motionless on floor (dead ball)
                if self.stationary_frames >= 4:
                    self.is_lost = True
                    self.is_stationary = True
                    self.tentative = None
                    self.kf = None
                    self.vx, self.vy = 0.0, 0.0
                    local_success = False
                else:
                    # Adaptive velocity lag: check for direction reversal (e.g. bounce / racket hit)
                    v_dot = (new_vx * self.vx + new_vy * self.vy)
                    eff_comp = 0.0 if v_dot < 0 else self.vel_comp
                    
                    self.vx = new_vx
                    self.vy = new_vy
                    self.last_x = dx
                    self.last_y = dy
                    self.tentative = None
                    
                    # Velocity lag compensation (compensates 3-frame convolution temporal centroid lag)
                    pred_x_512 = dx + eff_comp * new_vx
                    pred_y_512 = dy + eff_comp * new_vy
                    pred_pt_512 = (pred_x_512, pred_y_512)
                    pred_pt_orig = (pred_x_512 * scale_w, pred_y_512 * scale_h)
                    conf_out = conf
                    mode_str = "STATIONARY_HOLD" if self.is_stationary else "LOCAL_TRACK"
                    local_success = True
                    
                    if self.kf is None:
                        self.kf = init_kalman(dx, dy)
                    else:
                        self.kf.correct(np.array([[dx], [dy]], dtype=np.float32))
            else:
                local_success = False
        else:
            local_success = False

        # =========================================================================
        # Stage 2: Instant Coarse-to-Fine Global Recovery Mode
        # Seamlessly rescues tracking in the SAME frame whenever Local fails!
        # =========================================================================
        if not local_success:
            # Construct 256x144 temporal input from circular ring buffer
            self.buf_256[0, 0:3] = self.resized_slots[(self.slot_idx - 3) % 3]
            self.buf_256[0, 3:6] = self.resized_slots[(self.slot_idx - 2) % 3]
            self.buf_256[0, 6:9] = self.resized_slots[(self.slot_idx - 1) % 3]
            
            out_256 = self.trt_256.infer(self.buf_256).squeeze()
            gx, gy, conf_coarse = fast_argmax(out_256)
            sgx, sgy = subpixel_centroid(out_256, gx, gy, win=2)
            engine_str = "128+256" if not self.is_lost else "256x144"
            
            # Mechanism B: Adaptive Low-Contrast Proposal Rescue
            pass_coarse = False
            if conf_coarse >= self.t_coarse:
                pass_coarse = True
            elif self.adaptive_rescue and conf_coarse >= 0.08:
                # Compute Peak-to-Mean Ratio (PMR) to verify isolated candidate
                mean_bg = float(np.mean(out_256)) + 1e-6
                pmr = conf_coarse / mean_bg
                if pmr >= 3.2:
                    pass_coarse = True
                    
            if pass_coarse:
                cand_x = float(sgx) * 2.0
                cand_y = float(sgy) * 2.0
                
                # Native Fine Verification (128x128 Crop)
                v0, _ = robust_crop(self.frame_buffer[0], cand_x, cand_y, 64, 128)
                v1, _ = robust_crop(self.frame_buffer[1], cand_x, cand_y, 64, 128)
                v2, _ = robust_crop(self.frame_buffer[2], cand_x, cand_y, 64, 128)
                
                self.buf_128[0, 0:3] = np.transpose(v0, (2, 0, 1)) * self.norm_scale
                self.buf_128[0, 3:6] = np.transpose(v1, (2, 0, 1)) * self.norm_scale
                self.buf_128[0, 6:9] = np.transpose(v2, (2, 0, 1)) * self.norm_scale
                
                out_ver = self.trt_128.infer(self.buf_128).squeeze()
                vx_loc, vy_loc, conf_ver = fast_argmax(out_ver)
                svx, svy = subpixel_centroid(out_ver, vx_loc, vy_loc, win=2)
                engine_str += "+128_ver"
                
                accepted = False
                if conf_ver >= self.t_verify:
                    fine_x = cand_x - 64.0 + float(svx)
                    fine_y = cand_y - 64.0 + float(svy)
                    conf_out = conf_ver
                    mode_str = "INSTANT_RESCUE"
                    accepted = True
                elif not self.require_ver and conf_coarse >= 0.22 and conf_ver >= 0.16:
                    fine_x = cand_x - 64.0 + float(svx)
                    fine_y = cand_y - 64.0 + float(svy)
                    conf_out = conf_coarse
                    mode_str = "RESCUE_COARSE"
                    accepted = True
                    
                if accepted:
                    if self.confirm_rescue:
                        # Dynamic in-flight motion 2-hit confirmation
                        if self.tentative is not None:
                            tx, ty, tfid = self.tentative
                            jump = math.hypot(fine_x - tx, fine_y - ty)
                            if 2.0 <= jump <= self.jump_gate * 1.5:
                                # Confirmed valid moving shuttlecock!
                                self.last_x, self.last_y = fine_x, fine_y
                                self.vx = fine_x - tx
                                self.vy = fine_y - ty
                                self.is_lost = False
                                self.miss_count = 0
                                self.stationary_frames = 0
                                self.kf = init_kalman(fine_x, fine_y)
                                pred_pt_512 = (fine_x, fine_y)
                                pred_pt_orig = (fine_x * scale_w, fine_y * scale_h)
                                retro_pred = (tfid, (tx * scale_w, ty * scale_h))
                                self.tentative = None
                            else:
                                # Prior candidate was a static glitch, replace with new tentative
                                self.tentative = (fine_x, fine_y, target_mid_fid)
                        else:
                            # Register candidate for confirmation
                            self.tentative = (fine_x, fine_y, target_mid_fid)
                            self.last_x, self.last_y = fine_x, fine_y
                    else:
                        self.last_x, self.last_y = fine_x, fine_y
                        self.vx, self.vy = 0.0, 0.0
                        self.is_lost = False
                        self.miss_count = 0
                        self.stationary_frames = 0
                        self.kf = init_kalman(fine_x, fine_y)
                        pred_pt_512 = (fine_x, fine_y)
                        pred_pt_orig = (fine_x * scale_w, fine_y * scale_h)
                else:
                    self.miss_count += 1
                    if self.miss_count >= self.max_miss:
                        self.is_lost = True
                        self.tentative = None
                        self.kf = None
                        self.vx, self.vy = 0.0, 0.0
            else:
                self.miss_count += 1
                if self.miss_count >= self.max_miss:
                    self.is_lost = True
                    self.tentative = None
                    self.kf = None
                    self.vx, self.vy = 0.0, 0.0
                    
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.processed_frames += 1
        self.total_latency_ms += latency_ms
        
        telemetry = {
            "frame_id": target_mid_fid,
            "pred_point": pred_pt_orig,
            "retro_pred": retro_pred,
            "norm_point": pred_pt_512,
            "confidence": conf_out,
            "mode": mode_str,
            "latency_ms": latency_ms,
            "is_lost": self.is_lost,
            "crop_radius": crop_r,
            "velocity": (self.vx, self.vy),
            "speed": math.hypot(self.vx, self.vy),
            "engine_used": engine_str,
            "is_stationary": self.is_stationary,
            "rally_state": "PRE_SERVE_HOLD" if self.is_stationary else ("ACTIVE_RALLY" if not self.is_lost else "LOST")
        }
        
        return telemetry
