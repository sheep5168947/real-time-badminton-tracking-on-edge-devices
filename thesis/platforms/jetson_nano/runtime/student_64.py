# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import threading
from collections import deque
import pycuda.driver as cuda
import tensorrt as trt

class TRTModel:
    def __init__(self, engine_path, ctx):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
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

def fast_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

# 核心修改 1：極限視野縮小為 64
ROI_SIZE = 64

# Algorithm Parameters (Aligned with Thesis)
MOTION_THRESH = 10
CONF_THRESH_LOCAL = 0.55
CONF_THRESH_GLOBAL = 0.55
MAX_MISS_TOLERANCE = 3

# ==========================================
# Worker Thread: AI Inference (Hybrid Dual-Engine 64+512)
# ==========================================
def ai_thread(engine_64_path, engine_512_path):
    global shared_pred_point, shared_is_lost, shared_status, is_running, ai_ready
    print("[INFO] [AI Worker] TensorRT Initialization...")
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    
    try:
        # 核心修改 2：載入 64x64 引擎
        trt_64 = TRTModel(engine_64_path, cuda_ctx)
        trt_512 = TRTModel(engine_512_path, cuda_ctx)
    except Exception as e:
        print("[ERROR] TensorRT Init Failed: {}".format(e))
        is_running = False
        cuda_ctx.pop()
        return

    print("[INFO] [AI Worker] Performing Dummy Inference Warm-up...")
    # 核心修改 3：Dummy Warm-up 尺寸必須對應 64x64
    dummy_64 = np.zeros((1, 9, 64, 64), dtype=np.float32)
    dummy_512 = np.zeros((1, 9, 288, 512), dtype=np.float32)
    _ = trt_64(dummy_64)
    _ = trt_512(dummy_512)
    
    ai_ready = True
    print("[INFO] [AI Worker] Ready! Listening for frames.")
    
    ai_frame_queue = deque(maxlen=3)
    last_processed_id = -1
    
    # State Machine Variables
    is_lost = True
    miss_count = 0
    kf = None
    last_x_512, last_y_512 = W_TRAIN // 2, H_TRAIN // 2
    
    # 統計用變數，用於報表輸出
    stats_total_frames = 0
    stats_track_64 = 0
    stats_global_rec = 0
    
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
        
        # ==========================================
        # Phase 1: Local ROI Tracking (64x64)
        # ==========================================
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
            
            # 核心修改 4：使用 64x64 引擎推論
            raw_out = trt_64(tensor_x)
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
                current_status = "Track 64 (Conf:{:.2f})".format(conf)
                stats_track_64 += 1
            else:
                miss_count += 1
                
            if miss_count >= MAX_MISS_TOLERANCE:
                is_lost = True

        # ==========================================
        # Phase 2: Global Recovery (512x288)
        # ==========================================
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
                current_status = "Global Rec (Conf:{:.2f})".format(conf)
                stats_global_rec += 1
            else:
                current_status = "Global Search..."
                stats_global_rec += 1 # 搜尋中也算動用全域算力

        with bbox_lock:
            shared_pred_point = (last_x_512, last_y_512) if student_found else (0, 0)
            shared_is_lost = is_lost
            shared_status = current_status
            
        infer_time = (time.perf_counter() - start_time) * 1000
        if f_id % 30 == 0:
             print("[INFO] Frame {} | Status: {} | Latency: {:.2f} ms".format(f_id, current_status, infer_time))

    cuda_ctx.pop()
    print("\n========================================")
    print("📊 64x64 極限壓縮模式 - 引擎觸發佔比報告")
    print("總處理幀數: {}".format(stats_total_frames))
    if stats_total_frames > 0:
        print("[64x64 省電] 佔比: {:.1f}%".format((stats_track_64 / stats_total_frames) * 100))
        print("[全域救援] 佔比: {:.1f}%".format((stats_global_rec / stats_total_frames) * 100))
    print("========================================\n")
    print("[INFO] [AI Worker] Terminated.")

# ==========================================
# Main Thread: I/O and Rendering
# ==========================================
def main():
    global is_running, shared_frame_id, shared_frame_data
    print("[INFO] System Init (Extreme 64x64 Dual-TRT Mode)...")
    
    file_name = "1_01_00.mp4"
    video_path = "weights/{}".format(file_name)
    
    # 核心修改 5：指定你剛剛編譯好的 64x64 engine 路徑
    engine_64_path = "weights/miniroi_64x64.engine"
    engine_512_path = "weights/miniroi_512x288.engine"
    output_path = "weights/{}_student_64x64_nano.mp4".format(file_name)
    
    t_ai = threading.Thread(target=ai_thread, args=(engine_64_path, engine_512_path))
    t_ai.start()
    
    print("[INFO] [Main] Waiting for AI Engine to warm up...")
    while not ai_ready and is_running:
        time.sleep(0.1)
        
    print("[INFO] [Main] AI Ready. Starting Camera & Writer...")
    
    cap = cv2.VideoCapture(os.path.abspath(video_path))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W_ORIG, H_ORIG))
    
    fps = 30.0
    frame_delay = 1.0 / fps
    local_frame_id = 0
    
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
            # 這裡畫出來的綠色追蹤框，會因為 ROI_SIZE = 64 自動變成原本一半的大小
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
        cv2.putText(frame_resized, "System: Dual-TRT (64x64 + 512)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, "AI State: {}".format(status_text), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame_resized)
        local_frame_id += 1
        
        process_time = time.perf_counter() - start_time
        sleep_time = frame_delay - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    is_running = False
    cap.release()
    out.release()
    t_ai.join()
    print("[INFO] System Shutdown Complete. Saved to: {}".format(output_path))

if __name__ == "__main__":
    main()