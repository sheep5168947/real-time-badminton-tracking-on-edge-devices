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

# Shared Memory for Threading
shared_frame_id = -1
shared_frame_data = None
shared_box_center = (0, 0) 
shared_pred_point = (0, 0) 
shared_status = "Init"
frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True
ai_ready = False

W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288
ROI_SIZE = 128

# Set your start coordinates here (based on 512x288)
START_X, START_Y = 260, 140  

# ==========================================
# Worker Thread: Pure ROI Inference
# ==========================================
def ai_thread(engine_128_path):
    global shared_box_center, shared_pred_point, shared_status, is_running, ai_ready
    print("[INFO] [AI Worker] TensorRT Initialization...")
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    
    try:
        trt_128 = TRTModel(engine_128_path, cuda_ctx)
    except Exception as e:
        print("[ERROR] TensorRT Init Failed:", e)
        is_running = False
        cuda_ctx.pop()
        return

    print("[INFO] [AI Worker] Performing Dummy Inference Warm-up...")
    dummy_128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
    _ = trt_128(dummy_128)
    
    ai_ready = True
    print("[INFO] [AI Worker] Ready! Listening for frames.")
    
    ai_frame_queue = deque(maxlen=3)
    last_processed_id = -1
    
    last_x, last_y = START_X, START_Y
    
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
        
        # Disaster 1: No Kalman Filter
        pred_x, pred_y = last_x, last_y
        
        c0, _, _ = crop_and_pad(img0, pred_x, pred_y, ROI_SIZE)
        c1, _, _ = crop_and_pad(img1, pred_x, pred_y, ROI_SIZE)
        c2, _, _ = crop_and_pad(img2, pred_x, pred_y, ROI_SIZE)

        imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
        tensor_x = np.expand_dims(np.transpose(imgs, (2, 0, 1)), axis=0)
        
        raw_out = trt_128(tensor_x)
        heatmap = fast_sigmoid(raw_out).squeeze()
        
        # Disaster 2: No Motion Mask
        lx, ly, conf = get_shuttle_center_local(heatmap)
        
        if conf >= 0.5:
            last_x = pred_x - ROI_SIZE // 2 + lx
            last_y = pred_y - ROI_SIZE // 2 + ly
            current_status = "Tracking (Green Box)"
        else:
            # Disaster 3: No Global Recovery
            current_status = "LOST STUCK! (Red Box)"

        with bbox_lock:
            shared_box_center = (pred_x, pred_y)
            shared_pred_point = (last_x, last_y)
            shared_status = current_status
            
        infer_time = (time.perf_counter() - start_time) * 1000
        if f_id % 15 == 0:
             print("[INFO] Frame {} | Status: {} | Conf: {:.2f} | Latency: {:.2f} ms".format(f_id, current_status, conf, infer_time))

    cuda_ctx.pop()
    print("[INFO] [AI Worker] Terminated.")

# ==========================================
# Main Thread: I/O and Rendering
# ==========================================
def main():
    global is_running, shared_frame_id, shared_frame_data
    print("[INFO] System Init (Pure ROI Disaster Mode)...")
    
    video_path = "weights/test_video.mp4" 
    engine_128_path = "weights/miniroi_128x128.engine"
    output_path = "weights/disaster_pure_roi_trt.mp4"
    
    t_ai = threading.Thread(target=ai_thread, args=(engine_128_path,))
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
            box_cx, box_cy = shared_box_center
            px, py = shared_pred_point
            status_text = shared_status
            
        if box_cx != 0 and box_cy != 0:
            scale_x = W_ORIG / float(W_TRAIN)
            scale_y = H_ORIG / float(H_TRAIN)
            
            box_w = int(ROI_SIZE * scale_x)
            box_h = int(ROI_SIZE * scale_y)
            
            top_left = (int(box_cx * scale_x - box_w // 2), int(box_cy * scale_y - box_h // 2))
            bottom_right = (int(box_cx * scale_x + box_w // 2), int(box_cy * scale_y + box_h // 2))
            
            if "LOST" in status_text:
                cv2.rectangle(frame_resized, top_left, bottom_right, (0, 0, 255), 5)
            else:
                cv2.rectangle(frame_resized, top_left, bottom_right, (0, 255, 0), 2)
                draw_x = int(px * scale_x)
                draw_y = int(py * scale_y)
                cv2.circle(frame_resized, (draw_x, draw_y), 10, (0, 0, 255), -1)
            
        color = (0, 0, 255) if "LOST" in status_text else (0, 255, 0)
        cv2.putText(frame_resized, "Mode: Pure ROI (No KF, No Recovery)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, "Status: {}".format(status_text), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
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