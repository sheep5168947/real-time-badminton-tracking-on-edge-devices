# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import threading
import argparse
import math
import csv
from collections import deque
import pycuda.driver as cuda
import tensorrt as trt

# === 參數解析 (整合 ROI 切換與路徑彈性) ===
parser = argparse.ArgumentParser(description="Real-Time Asynchronous Tracking with GT Eval")
parser.add_argument("--roi_size", type=int, default=64, choices=[64, 128], help="ROI Size (64 or 128)")
parser.add_argument("--input", type=str, required=True, help="Input video path (e.g. raw/part1/video/match1_01.mp4)")
parser.add_argument("--csv", type=str, required=True, help="Input GT CSV path (e.g. raw/part1/csv/match1_01_ball.csv)")
parser.add_argument("--output", type=str, default="realtime_output.mp4", help="Output video path")
parser.add_argument("--motion", type=int, default=10, help="Motion Mask Threshold")
parser.add_argument("--local_conf", type=float, default=0.55, help="Local Confidence Threshold")
parser.add_argument("--global_conf", type=float, default=0.55, help="Global Confidence Threshold")
parser.add_argument("--miss", type=int, default=3, help="Max Miss Tolerance")
args = parser.parse_args()

ROI_SIZE = args.roi_size
MOTION_THRESH = args.motion
CONF_THRESH_LOCAL = args.local_conf
CONF_THRESH_GLOBAL = args.global_conf
MAX_MISS_TOLERANCE = args.miss

# === 智慧載入 Ground Truth ===
GT_DICT = {}
if os.path.exists(args.csv):
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            GT_DICT[int(row['Frame'])] = {'vis': int(row['Visibility']), 'x': float(row['X']), 'y': float(row['Y'])}
else:
    print(f"[警告] 找不到 {args.csv}，將無法計算 Recall 與 Precision！")

class TRTModel:
    def __init__(self, engine_path, ctx):
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
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def __call__(self, x_np):
        np.copyto(self.inputs[0]['host'], x_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy().reshape(self.outputs[0]['shape'])

def fast_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def get_shuttle_center_local(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0: return 0, 0, 0
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(idx[-1]), int(idx[-2]), float(max_val)

def crop_and_pad(img, center_x, center_y, size):
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = int(center_x - half), int(center_x + half)
    y_min, y_max = int(center_y - half), int(center_y + half)
    
    # [修復] 嚴格邊界防護，防止 out of bounds 當機
    valid_x_min = max(0, min(w, x_min))
    valid_x_max = max(0, min(w, x_max))
    valid_y_min = max(0, min(h, y_min))
    valid_y_max = max(0, min(h, y_max))
    
    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
    
    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, img.shape[2] if len(img.shape)==3 else 1), dtype=img.dtype)
        if cropped.size == 0:
            return padded, x_min, y_min
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        padded[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
        return padded, x_min, y_min
    return cropped, x_min, y_min

def init_kalman_filter(init_x, init_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
    return kf

# Shared Memory for Threading
shared_frame_id = -1
shared_frame_data = None
shared_pred_point = (0, 0)
shared_is_lost = True
shared_status = "Init"
frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True
ai_ready = False

W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288

# ==========================================
# Worker Thread: AI Inference (Hybrid Dual-Engine)
# ==========================================
def ai_thread(engine_local_path, engine_512_path):
    global shared_pred_point, shared_is_lost, shared_status, is_running, ai_ready
    print("[INFO] [AI Worker] TensorRT Initialization...")
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    
    try:
        trt_local = TRTModel(engine_local_path, cuda_ctx)
        trt_512 = TRTModel(engine_512_path, cuda_ctx)
    except Exception as e:
        print(f"[ERROR] TensorRT Init Failed: {e}")
        is_running = False
        cuda_ctx.pop()
        return

    print("[INFO] [AI Worker] Performing Dummy Inference Warm-up...")
    _ = trt_local(np.zeros((1, 9, ROI_SIZE, ROI_SIZE), dtype=np.float32))
    _ = trt_512(np.zeros((1, 9, 288, 512), dtype=np.float32))
    
    ai_ready = True
    print(f"[INFO] [AI Worker] Ready! Listening for frames. (Mode: {ROI_SIZE}x{ROI_SIZE})")
    
    ai_frame_queue = deque(maxlen=3)
    last_processed_id = -1
    
    is_lost = True
    miss_count = 0
    kf = None
    last_x_512, last_y_512 = W_TRAIN // 2, H_TRAIN // 2
    
    stats_total_frames = 0
    stats_track_local = 0
    stats_global_rec = 0
    TP, FP, FN = 0, 0, 0
    DIST_THRESH = 15.0
    
    scale_w = W_ORIG / float(W_TRAIN)
    scale_h = H_ORIG / float(H_TRAIN)
    ai_start_time = time.time()
    
    while is_running:
        with frame_lock:
            f_id = shared_frame_id
            f_data = shared_frame_data
            
        if f_id <= last_processed_id or f_data is None:
            time.sleep(0.001)
            continue
            
        last_processed_id = f_id
        start_time = time.perf_counter()
        
        f_train = cv2.resize(f_data, (W_TRAIN, H_TRAIN))
        ai_frame_queue.append(f_train)
        
        if len(ai_frame_queue) < 3:
            continue
            
        img0, img1, img2 = ai_frame_queue[0], ai_frame_queue[1], ai_frame_queue[2]
        student_found = False
        current_status = "Processing..."
        stats_total_frames += 1
        
        # Phase 1: Local ROI Tracking
        if not is_lost and kf is not None:
            predicted_state = kf.predict()
            pred_x, pred_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])
            
            c0, _, _ = crop_and_pad(img0, pred_x, pred_y, ROI_SIZE)
            c1, _, _ = crop_and_pad(img1, pred_x, pred_y, ROI_SIZE)
            c2, _, _ = crop_and_pad(img2, pred_x, pred_y, ROI_SIZE)
            
            g0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(cv2.absdiff(g1, g0), MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32) / 255.0

            imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
            tensor_x = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
            
            raw_out = trt_local(tensor_x)
            heatmap = fast_sigmoid(raw_out).squeeze()
            heatmap_masked = heatmap * cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            lx, ly, conf = get_shuttle_center_local(heatmap_masked)
            
            if conf >= CONF_THRESH_LOCAL:
                gx = pred_x - ROI_SIZE // 2 + lx
                gy = pred_y - ROI_SIZE // 2 + ly
                kf.correct(np.array([[gx], [gy]], dtype=np.float32))
                last_x_512, last_y_512 = gx, gy
                student_found = True
                miss_count = 0
                current_status = f"Track {ROI_SIZE} (Conf:{conf:.2f})"
                stats_track_local += 1
            else:
                miss_count += 1
                # 【新增】沒抓到球時，採用卡爾曼濾波的純預測座標，讓綠框繼續順滑跟隨！
                last_x_512, last_y_512 = pred_x, pred_y
                current_status = f"Coast {ROI_SIZE} (Miss:{miss_count})"
                
            if miss_count >= MAX_MISS_TOLERANCE:
                is_lost = True

        # Phase 2: Global Recovery
        if is_lost:
            g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(cv2.absdiff(g1, g0), MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32) / 255.0
            
            imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
            tensor_x = np.ascontiguousarray(np.transpose(imgs, (2, 0, 1))[np.newaxis, ...])
            
            raw_out = trt_512(tensor_x)
            heatmap = fast_sigmoid(raw_out).squeeze()
            heatmap_masked = heatmap * cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            gx, gy, conf = get_shuttle_center_local(heatmap_masked)

            if conf > CONF_THRESH_GLOBAL:
                kf = init_kalman_filter(gx, gy) 
                last_x_512, last_y_512 = gx, gy
                is_lost = False
                miss_count = 0
                student_found = True
                current_status = f"Global Rec (Conf:{conf:.2f})"
            else:
                current_status = "Global Search..."
            stats_global_rec += 1

        with bbox_lock:
            # 【修改】只要還沒徹底 lost，就繼續畫出座標 (不管是 AI 抓的還是 KF 預測的)
            shared_pred_point = (last_x_512, last_y_512) if not is_lost else (0, 0)
            shared_is_lost = is_lost
            shared_status = current_status
            
        # === 計算跳幀狀態下的真實 GT 誤差 ===
        if f_id in GT_DICT:
            gt_vis = GT_DICT[f_id]['vis']
            gt_x = GT_DICT[f_id]['x']
            gt_y = GT_DICT[f_id]['y']
            
            if student_found:
                pred_x_orig = last_x_512 * scale_w
                pred_y_orig = last_y_512 * scale_h
                if gt_vis == 1:
                    dist = math.hypot(pred_x_orig - gt_x, pred_y_orig - gt_y)
                    if dist <= DIST_THRESH:
                        TP += 1
                    else:
                        FP += 1
                        FN += 1
                else:
                    FP += 1
            else:
                if gt_vis == 1:
                    FN += 1

    ai_end_time = time.time()
    total_time = ai_end_time - ai_start_time
    avg_fps = stats_total_frames / total_time if total_time > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    cuda_ctx.pop()
    print("\n========================================")
    print(f"📊 {ROI_SIZE}x{ROI_SIZE} 實時跳幀模擬 - 最終報告")
    print(f"AI 實際處理幀數: {stats_total_frames}")
    print(f"AI 處理平均 FPS: {avg_fps:.2f}")
    if stats_total_frames > 0:
        print(f"[{ROI_SIZE}x{ROI_SIZE} 追蹤] 佔比: {(stats_track_local / stats_total_frames) * 100:.1f}%")
        print(f"[全域救援] 佔比: {(stats_global_rec / stats_total_frames) * 100:.1f}%")
    print(f"準確率 (Precision): {precision * 100:.2f}%")
    print(f"召回率 (Recall): {recall * 100:.2f}%")
    print("========================================\n")
    print("[INFO] [AI Worker] Terminated.")

# ==========================================
# Main Thread: I/O and Rendering
# ==========================================
def main():
    global is_running, shared_frame_id, shared_frame_data
    print(f"[INFO] System Init (Hybrid Async Mode: {ROI_SIZE}x{ROI_SIZE})...")
    
    if not os.path.exists(args.input):
        print(f"[致命錯誤] 找不到影片檔案: {args.input}")
        return

    engine_local_path = f"weights/miniroi_{ROI_SIZE}x{ROI_SIZE}.engine"
    engine_512_path = "weights/miniroi_512x288.engine"
    
    if not os.path.exists(engine_local_path) or not os.path.exists(engine_512_path):
        print("[致命錯誤] 找不到 TensorRT Engine，請確認 weights 目錄！")
        return
    
    t_ai = threading.Thread(target=ai_thread, args=(engine_local_path, engine_512_path))
    t_ai.start()
    
    print("[INFO] [Main] Waiting for AI Engine to warm up...")
    while not ai_ready and is_running:
        time.sleep(0.1)
        
    print("[INFO] [Main] AI Ready. Starting Camera Stream Simulation...")
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
        
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W_ORIG, H_ORIG))
    
    fps = 30.0
    frame_delay = 1.0 / fps
    local_frame_id = 0
    
    try:
        while is_running:
            start_time = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_resized = cv2.resize(frame, (W_ORIG, H_ORIG))
            
            with frame_lock:
                shared_frame_data = frame_resized.copy()
                shared_frame_id = local_frame_id
                
            with bbox_lock:
                px, py = shared_pred_point
                is_lost_state = shared_is_lost
                status_text = shared_status
                
            scale_x = W_ORIG / float(W_TRAIN)
            scale_y = H_ORIG / float(H_TRAIN)
            
            if is_lost_state:
                cv2.rectangle(frame_resized, (0, 0), (W_ORIG-1, H_ORIG-1), (0, 0, 255), 5)
            else:
                box_w = int(ROI_SIZE * scale_x)
                box_h = int(ROI_SIZE * scale_y)
                top_left = (int(px * scale_x - box_w // 2), int(py * scale_y - box_h // 2))
                bottom_right = (int(px * scale_x + box_w // 2), int(py * scale_y + box_h // 2))
                
                cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)
                if px != 0 and py != 0:
                    draw_x = int(px * scale_x)
                    draw_y = int(py * scale_y)
                    cv2.circle(frame_resized, (draw_x, draw_y), 10, (0, 0, 255), -1)
                    
            color = (0, 0, 255) if is_lost_state else (0, 255, 0)
            cv2.putText(frame_resized, f"System: Async Dual-TRT ({ROI_SIZE}x{ROI_SIZE})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_resized, f"AI State: {status_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            out.write(frame_resized)
            local_frame_id += 1
            
            # 主執行緒強制維持 30FPS，模擬真實相機輸入
            process_time = time.perf_counter() - start_time
            sleep_time = frame_delay - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n[警告] 收到手動中斷指令 (Ctrl+C)，正在安全關閉系統...")
    finally:
        is_running = False
        cap.release()
        out.release()
        t_ai.join()
        print(f"[INFO] System Shutdown Complete. Saved to: {args.output}")

if __name__ == "__main__":
    main()