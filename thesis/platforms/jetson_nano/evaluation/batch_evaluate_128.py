# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import argparse
import math
import csv
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt

parser = argparse.ArgumentParser(description="Dual-Engine Tracking 128x128 Batch Eval")
parser.add_argument("--motion", type=int, default=15, help="Motion Mask Threshold")
parser.add_argument("--local_conf", type=float, default=0.55, help="Local Confidence Threshold")
parser.add_argument("--global_conf", type=float, default=0.62, help="Global Confidence Threshold")
parser.add_argument("--miss", type=int, default=1, help="Max Miss Tolerance")
args = parser.parse_args()

MOTION_THRESH = args.motion
CONF_THRESH_LOCAL = args.local_conf
CONF_THRESH_GLOBAL = args.global_conf
MAX_MISS_TOLERANCE = args.miss

class TRTModel:
    def __init__(self, engine_path):
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

W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288
ROI_SIZE = 128

def main():
    engine_128_path = "weights/miniroi_128x128.engine"
    engine_512_path = "weights/miniroi_512x288.engine"
    
    trt_128 = TRTModel(engine_128_path)
    trt_512 = TRTModel(engine_512_path)
    
    dummy_128 = np.zeros((1, 9, 128, 128), dtype=np.float32)
    dummy_512 = np.zeros((1, 9, 288, 512), dtype=np.float32)
    _ = trt_128(dummy_128)
    _ = trt_512(dummy_512)
    
    base_dir = "raw/part1"
    video_files = []
    
    if not os.path.exists(base_dir):
        print(f"[致命錯誤] 找不到資料夾 {base_dir}")
        return
        
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "video":
            for file in files:
                if file.endswith(".mp4"):
                    vid_path = os.path.join(root, file)
                    match_dir = os.path.dirname(root)
                    csv_name = file.replace(".mp4", "_ball.csv")
                    csv_path = os.path.join(match_dir, "csv", csv_name)
                    if os.path.exists(csv_path):
                        video_files.append((vid_path, csv_path))
                        
    if len(video_files) == 0:
        print("[致命錯誤] 找不到任何有效的 video/csv 配對！")
        return

    stats_total_frames = 0
    stats_track_128 = 0
    stats_global_rec = 0
    TP, FP, FN = 0, 0, 0
    DIST_THRESH = 15.0
    scale_w = W_ORIG / float(W_TRAIN)
    scale_h = H_ORIG / float(H_TRAIN)
    ai_total_time = 0.0

    for vid_path, csv_path in video_files:
        GT_DICT = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                GT_DICT[int(row['Frame'])] = {'vis': int(row['Visibility']), 'x': float(row['X']), 'y': float(row['Y'])}
                
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
            
        frame_queue = []
        is_lost = True
        miss_count = 0
        kf = None
        last_x_512, last_y_512 = W_TRAIN // 2, H_TRAIN // 2
        local_frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            f_train = cv2.resize(frame, (W_TRAIN, H_TRAIN))
            frame_queue.append(f_train)
            if len(frame_queue) > 3:
                frame_queue.pop(0)
                
            if len(frame_queue) < 3:
                local_frame_id += 1
                continue
                
            img0, img1, img2 = frame_queue[0], frame_queue[1], frame_queue[2]
            student_found = False
            
            start_time = time.perf_counter()
            
            # Phase 1: Local ROI
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
                
                raw_out = trt_128(tensor_x)
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
                    stats_track_128 += 1
                else:
                    miss_count += 1
                    
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
                stats_global_rec += 1

            ai_total_time += (time.perf_counter() - start_time)
            stats_total_frames += 1

            if local_frame_id in GT_DICT:
                gt_vis = GT_DICT[local_frame_id]['vis']
                gt_x = GT_DICT[local_frame_id]['x']
                gt_y = GT_DICT[local_frame_id]['y']
                
                if student_found:
                    pred_x_orig = last_x_512 * scale_w
                    pred_y_orig = last_y_512 * scale_h
                    if gt_vis == 1:
                        if math.hypot(pred_x_orig - gt_x, pred_y_orig - gt_y) <= DIST_THRESH:
                            TP += 1
                        else:
                            FP += 1
                            FN += 1
                    else:
                        FP += 1
                else:
                    if gt_vis == 1:
                        FN += 1
                        
            local_frame_id += 1
        cap.release()
        
    avg_fps = stats_total_frames / ai_total_time if ai_total_time > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("\n========================================")
    print("📊 128x128 模式 - 引擎觸發與準確率報告")
    print("總處理幀數: {}".format(stats_total_frames))
    print("平均 FPS: {:.2f}".format(avg_fps))
    if stats_total_frames > 0:
        print("[128x128 省電] 佔比: {:.1f}%".format((stats_track_128 / stats_total_frames) * 100))
        print("[全域救援] 佔比: {:.1f}%".format((stats_global_rec / stats_total_frames) * 100))
    print("準確率 (Precision): {:.2f}%".format(precision * 100))
    print("召回率 (Recall): {:.2f}%".format(recall * 100))
    print("========================================\n")

if __name__ == "__main__":
    main()