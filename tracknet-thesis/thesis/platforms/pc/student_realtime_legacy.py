
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[3]))

import os
import cv2
import time
import torch
import numpy as np
import threading
from collections import deque

try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError:
    print("[ERROR] Cannot find student_model.py")

# ==========================================
# 核心工具函數
# ==========================================
def get_shuttle_center_local(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0: return 0, 0, 0
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y, max_val

def crop_and_pad(img, center_x, center_y, size):
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = int(center_x - half), int(center_x + half)
    y_min, y_max = int(center_y - half), int(center_y + half)

    valid_x_min, valid_x_max = max(0, x_min), min(w, x_max)
    valid_y_min, valid_y_max = max(0, y_min), min(h, y_max)

    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, img.shape[2] if len(img.shape)==3 else 1), dtype=img.dtype)
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

# ==========================================
# 全域共用變數與鎖 (Shared Memory & Locks)
# ==========================================
shared_frame_id = -1
shared_frame_data = None
shared_bbox = (0, 0)
shared_status = "Init"

frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True

# 系統常數
W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288
MOTION_THRESH = 15

# ==========================================
# 執行緒一：攝影機擷取 (Producer Thread)
# ==========================================
def camera_thread(video_path):
    global shared_frame_id, shared_frame_data, is_running
    print("[INFO] [Thread 1] Camera Producer Started.")
    
    cap = cv2.VideoCapture(video_path)
    fps = 30.0
    frame_delay = 1.0 / fps
    local_frame_id = 0
    
    while is_running:
        start_time = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            is_running = False
            break
            
        frame_resized = cv2.resize(frame, (W_ORIG, H_ORIG))
        
        # 寫入共用記憶體
        with frame_lock:
            shared_frame_data = frame_resized.copy()
            shared_frame_id = local_frame_id
            
        local_frame_id += 1
        
        # 模擬真實攝影機的 30 FPS 硬體物理限制
        read_time = time.perf_counter() - start_time
        sleep_time = frame_delay - read_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    print("[INFO] [Thread 1] Camera Producer Terminated.")

# ==========================================
# 執行緒二：AI 推論 (Consumer Thread)
# ==========================================
def ai_thread(weights_path):
    global shared_bbox, shared_status, is_running
    print("[INFO] [Thread 2] AI Inference Started (PyTorch Backend).")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    ai_frame_queue = deque(maxlen=3)
    last_processed_id = -1
    
    is_lost = True
    kf = None
    last_x_512, last_y_512 = W_TRAIN//2, H_TRAIN//2
    
    while is_running:
        # 1. 讀取最新畫面
        with frame_lock:
            f_id = shared_frame_id
            f_data = shared_frame_data
            
        # 等待新畫面 (避免重複推論同一張)
        if f_id <= last_processed_id or f_data is None:
            time.sleep(0.001)
            continue
            
        last_processed_id = f_id
        start_time = time.perf_counter()
        
        # 轉為模型尺寸並存入佇列
        f_train = cv2.resize(f_data, (W_TRAIN, H_TRAIN))
        ai_frame_queue.append(f_train)
        
        if len(ai_frame_queue) < 3:
            continue
            
        img0, img1, img2 = ai_frame_queue[0], ai_frame_queue[1], ai_frame_queue[2]
        student_found = False
        current_status = "Lost"
        
        # ==========================================
        # 嚴格固定 ROI 演算法邏輯
        # ==========================================
        if not is_lost and kf is not None:
            predicted_state = kf.predict()
            pred_x, pred_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])
            current_size = 128
            thresh = 0.62

            c0, _, _ = crop_and_pad(img0, pred_x, pred_y, current_size)
            c1, _, _ = crop_and_pad(img1, pred_x, pred_y, current_size)
            c2, _, _ = crop_and_pad(img2, pred_x, pred_y, current_size)
            
            g0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g0)
            _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32) / 255.0

            imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
            tensor_x = torch.from_numpy(np.transpose(imgs, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                heatmap = torch.sigmoid(model(tensor_x)).squeeze().cpu().numpy()
            
            heatmap_masked = heatmap * cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            lx, ly, conf = get_shuttle_center_local(heatmap_masked)
            
            if conf >= thresh:
                gx = pred_x - current_size // 2 + lx
                gy = pred_y - current_size // 2 + ly
                kf.correct(np.array([[gx], [gy]], dtype=np.float32))
                last_x_512, last_y_512 = gx, gy
                student_found = True
                current_status = f"Track 128 (Conf:{conf:.2f})"
            else:
                is_lost = True

        if is_lost:
            g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g1, g0)
            _, motion_mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.float32) / 255.0

            imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
            tensor_x = torch.from_numpy(np.transpose(imgs, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                heatmap = torch.sigmoid(model(tensor_x)).squeeze().cpu().numpy()

            heatmap_masked = heatmap * cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            heatmap_masked[:60, :] = 0.0 
            
            gx, gy, conf = get_shuttle_center_local(heatmap_masked)

            if conf > 0.55:
                kf = init_kalman_filter(gx, gy) 
                last_x_512, last_y_512 = gx, gy
                is_lost = False
                student_found = True
                current_status = f"Global Rec (Conf:{conf:.2f})"
            else:
                current_status = "Searching..."

        # 2. 將算好的座標映射回 1280 空間並寫入共用記憶體
        if student_found:
            final_x = int(last_x_512 * (W_ORIG / W_TRAIN))
            final_y = int(last_y_512 * (H_ORIG / H_TRAIN))
        else:
            final_x, final_y = 0, 0 # 代表未找到

        with bbox_lock:
            shared_bbox = (final_x, final_y)
            shared_status = current_status
            
        infer_time = (time.perf_counter() - start_time) * 1000
        if f_id % 30 == 0:
             print(f"[INFO] [Thread 2] Frame {f_id} | Mode: {current_status} | Latency: {infer_time:.2f} ms")

    print("[INFO] [Thread 2] AI Inference Terminated.")

# ==========================================
# 執行緒三：渲染與寫入影片 (Main Display Thread)
# ==========================================
def main():
    global is_running
    print("[INFO] System Initialization...")
    
    video_path = r"dataset\raw\part1\match1\video\1_01_00.mp4" 
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 
    output_path = "student_realtime_success.mp4"
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W_ORIG, H_ORIG))
    
    t1 = threading.Thread(target=camera_thread, args=(video_path,))
    t2 = threading.Thread(target=ai_thread, args=(weights_path,))
    
    t1.start()
    t2.start()
    
    print("[INFO] [Thread 3] Display Writer Started.")
    time.sleep(1.0) # 等待攝影機預熱
    
    fps = 30.0
    frame_delay = 1.0 / fps
    last_drawn_id = -1
    
    while is_running:
        start_time = time.perf_counter()
        
        with frame_lock:
            f_id = shared_frame_id
            f_data = shared_frame_data
            
        if f_data is None:
            continue
            
        if f_id <= last_drawn_id:
            # 避免同一幀畫兩次，這才是真正的 Real-time 節奏控制
            time.sleep(0.001)
            continue
            
        last_drawn_id = f_id
        display_frame = f_data.copy()
        
        # 讀取非同步的 AI 座標
        with bbox_lock:
            cx, cy = shared_bbox
            status_text = shared_status
            
        # 繪圖
        if cx != 0 and cy != 0:
            cv2.circle(display_frame, (cx, cy), 12, (0, 0, 255), -1)
            
        color = (0, 255, 0) if "Track" in status_text or "Rec" in status_text else (0, 0, 255)
        cv2.putText(display_frame, f"System: Async PyTorch", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"AI State: {status_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(display_frame)
        
        process_time = time.perf_counter() - start_time
        sleep_time = frame_delay - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    out.release()
    t1.join()
    t2.join()
    print(f"[INFO] System Shutdown Complete. Output saved to: {output_path}")

if __name__ == "__main__":
    main()