
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import os
import glob
import csv
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys

try:
    from tracknet.models.tracknet_v4 import TrackNetV4
except ImportError:
    print("Error: Cannot find tracknet module. Please ensure this script is in the tracknet-series-pytorch root directory.")
    sys.exit(1)

def crop_and_pad(img, center_x, center_y, size=128):
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half
    
    valid_x_min, valid_x_max = max(0, x_min), min(w, x_max)
    valid_y_min, valid_y_max = max(0, y_min), min(h, y_max)
    
    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
    
    if cropped.shape[0] != size or cropped.shape[1] != size:
        if len(img.shape) == 3:
            padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((size, size), dtype=img.dtype)
            
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        padded[offset_y:offset_y+cropped.shape[0], offset_x:offset_x+cropped.shape[1]] = cropped
        return padded
    return cropped

def process_and_cache_v4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Offline Distillation Data Preparation. Device: {device}")

    print("Loading TrackNetV4 Teacher Model...")
    teacher_model = TrackNetV4().to(device)
    
    # 確保這裡指向你剛訓練出來的 V4 王牌導師！
    weights_path = r"outputs\train\tracknet_v4_20260715_075947\checkpoints\best.pt" 
    if not os.path.exists(weights_path):
        print(f"Error: Cannot find weights file at {weights_path}")
        return

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            teacher_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            teacher_model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error during weight loading: {e}")
        return

    teacher_model.eval()
    print("TrackNetV4 Teacher Model loaded successfully.")

    # === 核心解封印區塊 ===
    raw_root = os.path.join("dataset", "raw", "part1")
    output_dir = os.path.join("dataset", "cached_pt_v4")
    os.makedirs(output_dir, exist_ok=True)

    # 使用萬用字元 (*) 深入尋找所有 match 資料夾中的 CSV
    csv_files = glob.glob(os.path.join(raw_root, "*", "csv", "*_ball.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {raw_root}.")
        return
        
    print(f"\n🎯 總共找到 {len(csv_files)} 個影片片段準備進行蒸餾！\n")

    for csv_file in csv_files:
        video_name = os.path.basename(csv_file).replace("_ball.csv", "")
        
        # 動態抓取這是哪一場比賽 (例如: match1, match2)
        match_dir = os.path.dirname(os.path.dirname(csv_file))
        match_name = os.path.basename(match_dir)
        
        video_file = os.path.join(match_dir, "video", f"{video_name}.mp4")
        
        # 快取檔名加上 match 名稱，防止覆蓋
        save_path = os.path.join(output_dir, f"{match_name}_{video_name}.pt")

        if not os.path.exists(video_file):
            print(f"Warning: Video file not found: {video_file}")
            continue

        print(f"\nProcessing Video (Overwriting if exists): [{match_name}] {video_name}")
        cap = cv2.VideoCapture(video_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (512, 288))
            frames.append(frame)
        cap.release()

        cached_data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            keys = reader.fieldnames
            vis_key = 'visible' if 'visible' in keys else 'Visibility'
            frame_key = 'frame_num' if 'frame_num' in keys else 'Frame'
            x_key = 'x' if 'x' in keys else 'X'
            y_key = 'y' if 'y' in keys else 'Y'

            samples = [row for row in reader if int(row[vis_key]) == 1]
            
            for row in tqdm(samples, desc="Generating Teacher Heatmaps"):
                f_num = int(row[frame_key])
                
                if f_num < 1 or f_num >= len(frames) - 1:
                    continue
                
                c_x = int(float(row[x_key]) * (512 / 1280.0))
                c_y = int(float(row[y_key]) * (288 / 720.0))
                
                img0 = frames[f_num - 1]
                img1 = frames[f_num]
                img2 = frames[f_num + 1]

                tensor0 = torch.from_numpy(img0).permute(2, 0, 1).float() / 255.0
                tensor1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
                tensor2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
                input_tensor = torch.cat([tensor0, tensor1, tensor2], dim=0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    teacher_output = teacher_model(input_tensor)
                    if teacher_output.min() < 0: 
                        teacher_heatmap = torch.sigmoid(teacher_output)
                    else:
                        teacher_heatmap = teacher_output
                    
                    if teacher_heatmap.shape[1] == 3:
                        heatmap_np = teacher_heatmap[0, 1].cpu().numpy()
                    else:
                        heatmap_np = teacher_heatmap[0, 0].cpu().numpy()
                
                crop0 = crop_and_pad(img0, c_x, c_y, size=128)
                crop1 = crop_and_pad(img1, c_x, c_y, size=128)
                crop2 = crop_and_pad(img2, c_x, c_y, size=128)
                crop_heatmap = crop_and_pad(heatmap_np, c_x, c_y, size=128)

                crop_tensor0 = torch.from_numpy(crop0).permute(2, 0, 1).float() / 255.0
                crop_tensor1 = torch.from_numpy(crop1).permute(2, 0, 1).float() / 255.0
                crop_tensor2 = torch.from_numpy(crop2).permute(2, 0, 1).float() / 255.0
                
                student_input = torch.cat([crop_tensor0, crop_tensor1, crop_tensor2], dim=0)
                student_target = torch.from_numpy(crop_heatmap).unsqueeze(0).float()

                cached_data.append({
                    "input": student_input,      
                    "target": student_target     
                })

        if cached_data:
            torch.save(cached_data, save_path)
            # print 不再擠在一起，看起來更簡潔
        else:
            print(f"Warning: No valid data generated for {video_name}.")

if __name__ == '__main__':
    process_and_cache_v4()