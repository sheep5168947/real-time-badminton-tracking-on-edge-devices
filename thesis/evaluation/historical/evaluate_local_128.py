
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[3]))

import os
import cv2
import csv
import torch
import numpy as np
import math
from tqdm import tqdm
from collections import deque

try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError:
    print("[ERROR] Cannot find student_model.py")

# ==========================================
# Core Utils
# ==========================================
def get_shuttle_center_local(heatmap):
    max_val = np.max(heatmap)
    if max_val == 0:
        return 0, 0, 0
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y, max_val

def crop_and_pad(img, center_x, center_y, size):
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = int(center_x - half), int(center_x + half)
    y_min, y_max = int(center_y - half), int(center_y + half)

    # 🛑 防呆極限保護：強制鉗制邊界，防止 Python 的負數 Index 切片！
    valid_x_min = max(0, min(w, x_min))
    valid_x_max = max(0, min(w, x_max))
    valid_y_min = max(0, min(h, y_min))
    valid_y_max = max(0, min(h, y_max))

    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, img.shape[2] if len(img.shape)==3 else 1), dtype=img.dtype)
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        
        # 若預測框徹底飛出畫面 (裁切面積為 0)，則直接回傳全黑影像
        if cropped.size > 0:
            padded[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
        return padded, x_min, y_min
        
    return cropped, x_min, y_min

def init_kalman_filter(init_x, init_y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePost = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
    return kf

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ==========================================
# Main Evaluation
# ==========================================
def evaluate_pure_128():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路徑設定
    video_path = r"dataset\raw\part1\match1\video\1_01_00.mp4" 
    csv_path = r"dataset\raw\part1\match1\csv\1_01_00_ball.csv"
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 

    if not os.path.exists(video_path):
        print("[ERROR] Video not found.")
        return

    # 1. 載入 Ground Truth
    ground_truth = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        keys = reader.fieldnames
        f_key = 'frame_num' if 'frame_num' in keys else 'Frame'
        v_key = 'visible' if 'visible' in keys else 'Visibility'
        x_key, y_key = ('x', 'y') if 'x' in keys else ('X', 'Y')
        for row in reader:
            if int(row[v_key]) == 1:
                ground_truth[int(row[f_key])] = (float(row[x_key]), float(row[y_key]))

    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    W_orig, H_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    W_train, H_train = 512, 288 
    MOTION_THRESH = 15
    
    frame_queue = deque(maxlen=3)
    predictions = {}
    
    # 尋找第一個有球的 Frame 作為神諭初始化點
    first_gt_frame = min(ground_truth.keys())
    init_x_orig, init_y_orig = ground_truth[first_gt_frame]
    init_x_512 = int(init_x_orig * (W_train / float(W_orig)))
    init_y_512 = int(init_y_orig * (H_train / float(H_orig)))

    print(f"[INFO] Oracle Initialization at Frame {first_gt_frame} -> ({init_x_512}, {init_y_512})")
    
    # 初始化卡爾曼濾波器
    kf = init_kalman_filter(init_x_512, init_y_512)
    
    # 快轉影片到 first_gt_frame - 2，開始填充滑動視窗
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_gt_frame - 2)
    for _ in range(2):
        ret, frame = cap.read()
        if not ret: break
        frame_queue.append(cv2.resize(frame, (W_train, H_train)))

    frame_idx = first_gt_frame
    
    # 從 first_gt_frame 開始跑到影片結束
    total_frames_to_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - first_gt_frame
    
    print("[INFO] Starting Pure 128x128 Tracking Ablation Study...")
    with tqdm(total=total_frames_to_process) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            f_resized = cv2.resize(frame, (W_train, H_train))
            frame_queue.append(f_resized)
            
            img0, img1, img2 = frame_queue[0], frame_queue[1], frame_queue[2]
            
            # KF 預測當前可能的中心點
            predicted_state = kf.predict()
            pred_x, pred_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])
            current_size = 128
            thresh = 0.55

            # 直接進行 128x128 裁切 (永不退回 512 全域搜索)
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
                
                # 紀錄座標
                pred_x_orig = gx * (W_orig / float(W_train))
                pred_y_orig = gy * (H_orig / float(H_train))
                predictions[frame_idx] = (pred_x_orig, pred_y_orig)
            else:
                # 若信心度不足，純 128 系統沒有 512 引擎可救援
                # 只能選擇放棄預測，任由 KF 盲推 (這會導致嚴重的漏抓 / FN)
                pass

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # === 計算成績 ===
    print("\n" + "="*50)
    print("Ablation Study: Pure 128x128 Tracker Evaluation")
    print("="*50)
    print(f"Ground Truth Count (GT): {len(ground_truth)}")
    print(f"Prediction Count (Pred): {len(predictions)}")
    print("-" * 50)
    
    thresholds = [15.0, 20.0, 30.0, 50.0]
    
    for thresh in thresholds:
        TP, FP, FN = 0, 0, 0
        all_frames = set(ground_truth.keys()).union(set(predictions.keys()))
        
        for f in all_frames:
            has_gt = f in ground_truth
            has_pred = f in predictions
            
            if has_gt and has_pred:
                dist = calculate_distance(ground_truth[f], predictions[f])
                if dist <= thresh:
                    TP += 1
                else:
                    FP += 1
                    FN += 1
            elif has_pred and not has_gt:
                FP += 1
            elif has_gt and not has_pred:
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Distance Threshold = {thresh} pixels:")
        print(f"TP: {TP} | FP: {FP} | FN: {FN}")
        print(f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1: {f1*100:.2f}%")
        print("-" * 50)

if __name__ == '__main__':
    evaluate_pure_128()