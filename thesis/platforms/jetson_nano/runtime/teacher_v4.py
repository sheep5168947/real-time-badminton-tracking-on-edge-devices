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

def get_shuttle_center(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0: return 0, 0, 0
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(idx[-1]), int(idx[-2]), float(max_val)

shared_frame_id = -1
shared_frame_data = None
shared_bbox = (0, 0)
shared_status = "Init"
frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True
ai_ready = False

W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288

def ai_thread(engine_path):
    global shared_bbox, shared_status, is_running, ai_ready
    print("[INFO] [AI Worker] TensorRT Initialization...")
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    
    try:
        trt_model = TRTModel(engine_path, cuda_ctx)
    except Exception as e:
        print("[ERROR] TensorRT Init Failed:", e)
        is_running = False
        cuda_ctx.pop()
        return

    # Dummy Warm-up
    print("[INFO] [AI Worker] Performing Dummy Inference Warm-up...")
    dummy_input = np.zeros((1, 9, 288, 512), dtype=np.float32)
    _ = trt_model(dummy_input)
    
    ai_ready = True
    print("[INFO] [AI Worker] Ready! Listening for frames.")
    
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
            
        img0, img1, img2 = ai_frame_queue[0], ai_frame_queue[1], ai_frame_queue[2]
        imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
        tensor_x = np.expand_dims(np.transpose(imgs, (2, 0, 1)), axis=0)
        
        raw_out = trt_model(tensor_x)
        heatmap = fast_sigmoid(raw_out).squeeze()
        lx, ly, conf = get_shuttle_center(heatmap)
        
        if conf > 0.5:
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
        print(f"[INFO] [AI Worker] V4 Inference #{inference_count} | Latency: {infer_time:.2f} ms")

    cuda_ctx.pop()
    print("[INFO] [AI Worker] Terminated.")

def main():
    global is_running, shared_frame_id, shared_frame_data
    print("[INFO] System Init (Teacher 2-Thread Architecture)...")
    
    video_path = "weights/test_video.mp4" 
    engine_path = "weights/tracknet_v4.engine"
    output_path = "weights/teacher_realtime_nano.mp4"
    
    t_ai = threading.Thread(target=ai_thread, args=(engine_path,))
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
            cx, cy = shared_bbox
            status_text = shared_status
            
        if cx != 0 and cy != 0:
            cv2.circle(frame_resized, (cx, cy), 12, (0, 0, 255), -1)
            
        color = (0, 255, 0) if "Detect" in status_text else (0, 0, 255)
        cv2.putText(frame_resized, "System: Teacher V4 (TensorRT)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, f"AI State: {status_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
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
    print(f"[INFO] System Shutdown Complete. Saved to: {output_path}")

if __name__ == "__main__":
    main()