
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

    valid_x_min, valid_x_max = max(0, x_min), min(w, x_max)
    valid_y_min, valid_y_max = max(0, y_min), min(h, y_max)

    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, img.shape[2] if len(img.shape)==3 else 1), dtype=img.dtype)
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        if cropped.size > 0:
            padded[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
        return padded, x_min, y_min
    return cropped, x_min, y_min

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def evaluate_no_kf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    video_path = r"dataset\raw\part1\match1\video\1_01_00.mp4" 
    csv_path = r"dataset\raw\part1\match1\csv\1_01_00_ball.csv"
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 

    if not os.path.exists(video_path):
        print("[ERROR] Video not found.")
        return

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    W_train, H_train = 512, 288 
    MOTION_THRESH = 15
    
    frame_queue = deque(maxlen=3)
    predictions = {}
    
    for _ in range(2):
        ret, frame = cap.read()
        if not ret: break
        frame_queue.append(cv2.resize(frame, (W_train, H_train)))

    frame_idx = 2
    
    # 💡 拔除 KF，改用純變數紀錄上一幀座標
    is_lost = True
    last_x_512, last_y_512 = 0, 0
    
    print("[INFO] Starting Ablation Study: Dual-Engine WITHOUT Kalman Filter...")
    with tqdm(total=total_frames-2) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            f_resized = cv2.resize(frame, (W_train, H_train))
            frame_queue.append(f_resized)
            
            img0, img1, img2 = frame_queue[0], frame_queue[1], frame_queue[2]
            student_found = False
            final_x_512, final_y_512 = 0, 0

            # ==========================================
            # 模式 A: 128x128 追蹤 (無 KF 導航)
            # ==========================================
            if not is_lost:
                # 只能死盯著上一幀的位置，無法預測未來
                pred_x, pred_y = last_x_512, last_y_512
                current_size = 128
                thresh = 0.55

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
                    last_x_512, last_y_512 = gx, gy
                    final_x_512, final_y_512 = gx, gy
                    student_found = True
                else:
                    is_lost = True

            # ==========================================
            # 模式 B: 512x288 全域搜索
            # ==========================================
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

                if conf > 0.62:
                    last_x_512, last_y_512 = gx, gy 
                    final_x_512, final_y_512 = gx, gy
                    is_lost = False
                    student_found = True

            # ==========================================
            # 紀錄座標
            # ==========================================
            if student_found:
                pred_x_orig = final_x_512 * (W_orig / float(W_train))
                pred_y_orig = final_y_512 * (H_orig / float(H_train))
                predictions[frame_idx] = (pred_x_orig, pred_y_orig)
            
            frame_idx += 1
            pbar.update(1)

    cap.release()

    # === 計算成績 ===
    print("\n" + "="*50)
    print("Ablation Study: Dual-Engine WITHOUT Kalman Filter")
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
                if dist <= thresh: TP += 1
                else: FP += 1; FN += 1
            elif has_pred and not has_gt: FP += 1
            elif has_gt and not has_pred: FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Distance Threshold = {thresh} pixels:")
        print(f"TP: {TP} | FP: {FP} | FN: {FN}")
        print(f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1: {f1*100:.2f}%")
        print("-" * 50)

if __name__ == '__main__':
    evaluate_no_kf()