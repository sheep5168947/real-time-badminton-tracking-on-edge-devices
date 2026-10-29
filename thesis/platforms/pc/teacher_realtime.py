
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
    # 請根據你的專案結構引入 V4 教師模型
    from tracknet.models.tracknet_v4 import TrackNetV4
except ImportError:
    print("[ERROR] Cannot find TrackNetV4_Model.py")

# ==========================================
# 核心工具函數
# ==========================================
def get_shuttle_center(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0: 
        return 0, 0, 0
        
    # 取得最大值的多維索引 (可能是 2D, 3D 或 4D)
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # 無論維度多少，空間座標 Y 和 X 永遠在最後兩個維度
    y, x = idx[-2], idx[-1] 
    
    return int(x), int(y), float(max_val)

# ==========================================
# 全域共用變數與鎖
# ==========================================
shared_frame_id = -1
shared_frame_data = None
shared_bbox = (0, 0)
shared_status = "Init"

frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True

W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288

# ==========================================
# 執行緒一：攝影機擷取 (Producer)
# ==========================================
def camera_thread(video_path):
    global shared_frame_id, shared_frame_data, is_running
    print("[INFO] [Thread 1] Camera Producer Started (30 FPS).")
    
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
        
        with frame_lock:
            shared_frame_data = frame_resized.copy()
            shared_frame_id = local_frame_id
            
        local_frame_id += 1
        
        read_time = time.perf_counter() - start_time
        sleep_time = frame_delay - read_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    print("[INFO] [Thread 1] Camera Producer Terminated.")

# ==========================================
# 執行緒二：AI 推論 - Teacher V4 (Consumer)
# ==========================================
def ai_thread(weights_path):
    global shared_bbox, shared_status, is_running
    print("[INFO] [Thread 2] Teacher AI Inference Started (PyTorch).")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrackNetV4().to(device) # 初始化 V4
    # 1. 先把綜合大禮包讀取出來
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    # 2. 從大禮包中拿出真正的模型權重，再餵給模型
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    ai_frame_queue = deque(maxlen=3)
    last_processed_id = -1
    inference_count = 0
    
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
            
        # Teacher 是全圖預測，直接將 3 幀 512x288 送入
        img0, img1, img2 = ai_frame_queue[0], ai_frame_queue[1], ai_frame_queue[2]
        imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
        tensor_x = torch.from_numpy(np.transpose(imgs, (2, 0, 1))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            heatmap = torch.sigmoid(model(tensor_x)).squeeze().cpu().numpy()
            
        # V4 預設是物件偵測器，無狀態機，直接取全圖最大值
        lx, ly, conf = get_shuttle_center(heatmap)
        
        if conf > 0.6:
            # V4 直接輸出 512x288 空間的座標
            final_x = int(lx * (W_ORIG / W_TRAIN))
            final_y = int(ly * (H_ORIG / H_TRAIN))
            current_status = f"V4 Detect (Conf:{conf:.2f})"
        else:
            final_x, final_y = 0, 0
            current_status = "No Ball"

        with bbox_lock:
            shared_bbox = (final_x, final_y)
            shared_status = current_status
            
        infer_time = (time.perf_counter() - start_time) * 1000
        inference_count += 1
        
        if inference_count % 15 == 0:
             print(f"[INFO] [Thread 2] V4 Inference #{inference_count} | Latency: {infer_time:.2f} ms")

    print("[INFO] [Thread 2] Teacher AI Inference Terminated.")

# ==========================================
# 執行緒三：渲染與寫入影片 (Main Display)
# ==========================================
def main():
    global is_running
    print("[INFO] System Initialization (Teacher Real-time)...")
    
    video_path = r"dataset\raw\part1\match1\video\1_01_00.mp4" 
    # 請替換為你訓練好的 Teacher (V4) 權重路徑
    weights_path = r"outputs\train\tracknet_v4_20260715_075947\checkpoints\best.pt"
    output_path = "teacher_realtime_pc.mp4"
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Cannot find Teacher weights: {weights_path}")
        return

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W_ORIG, H_ORIG))
    
    t1 = threading.Thread(target=camera_thread, args=(video_path,))
    t2 = threading.Thread(target=ai_thread, args=(weights_path,))
    
    t1.start()
    t2.start()
    
    print("[INFO] [Thread 3] Display Writer Started.")
    time.sleep(1.0) 
    
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
            time.sleep(0.001)
            continue
            
        last_drawn_id = f_id
        display_frame = f_data.copy()
        
        with bbox_lock:
            cx, cy = shared_bbox
            status_text = shared_status
            
        if cx != 0 and cy != 0:
            cv2.circle(display_frame, (cx, cy), 12, (0, 0, 255), -1)
            
        color = (0, 255, 0) if "Detect" in status_text else (0, 0, 255)
        cv2.putText(display_frame, "System: Teacher V4 (PC PyTorch)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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