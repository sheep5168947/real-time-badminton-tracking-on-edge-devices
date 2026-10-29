
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm

from thesis.models.miniroi_unet import MiniROIUNet

# === 終極武器：賞罰分明加權均方誤差損失函數 ===
class WeightedMSELoss(nn.Module):
    def __init__(self, ball_weight=10.0, false_positive_penalty=5.0):
        super(WeightedMSELoss, self).__init__()
        self.ball_weight = ball_weight
        self.false_positive_penalty = false_positive_penalty

    def forward(self, pred, target):
        base_loss = (pred - target) ** 2
        
        # 建立一個全為 1.0 的基礎權重遮罩
        weight_mask = torch.ones_like(target)
        
        # 1. 賞（抓漏球）：如果這裡是球，權重放大 10 倍
        weight_mask[target > 0] = self.ball_weight
        
        # 2. 罰 (抓假高分)：如果是背景 (target == 0)，但模型卻預測超過 0.1 的亮度
        # 代表它在亂認背景特徵，直接給予 5 倍的嚴厲懲罰
        false_positive_mask = (target == 0) & (pred > 0.1)
        weight_mask[false_positive_mask] = self.false_positive_penalty
        
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

# === 資料集讀取 (V4 新版) ===
class CachedVideoDataset(Dataset):
    def __init__(self, pt_file):
        # 載入 V4 產生的新版快取檔 (包含 input 與 target)
        # 加上 weights_only=False 解決 PyTorch 2.6 安全性限制
        self.data = torch.load(pt_file, weights_only=False)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # input: (9, 128, 128), target: (1, 128, 128)
        return item["input"], item["target"]

# === 舊版資料集讀取 (保留備用) ===
# class CachedVideoDatasetOld(Dataset):
#     def __init__(self, pt_file):
#         self.data = torch.load(pt_file, weights_only=False)
#         
#     def __len__(self):
#         return len(self.data)
#         
#     def __getitem__(self, idx):
#         return self.data[idx]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Device: {device}")
    
    # 1. 建立模型
    model = MiniROIUNet().to(device)
    
    # 2. 載入權重或從頭訓練
    base_weights = "checkpoints/epoch_30.pth"
    if os.path.exists(base_weights):
        # 若未來需要讀取舊權重，加上 weights_only=False 放行
        model.load_state_dict(torch.load(base_weights, weights_only=False))
        print(f"Successfully loaded base weights from {base_weights}. Starting fine-tuning.")
    else:
        print("Base weights not found. Starting training from scratch with random initialization.")
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedMSELoss()
    
    # 3. 指定 V4 產生的快取資料夾
    video_pt_files = glob.glob("dataset/cached_pt_v4/*.pt")
    
    if not video_pt_files:
        print("Error: No .pt files found in dataset/cached_pt_v4/")
        return
        
    epochs = 30
    batch_size = 16
    save_dir = "checkpoints_v4_student"
    os.makedirs(save_dir, exist_ok=True) 
    
    print("\n--- Starting Weighted Training ---")
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0.0
        total_batches = 0
        
        for pt_file in video_pt_files:
            video_name = os.path.basename(pt_file)
            print(f"\nLoading cache: {video_name} ...")
            dataset = CachedVideoDataset(pt_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [{video_name}]", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
                pbar.set_postfix({'w_Loss': f"{loss.item():.5f}"})
                
        if total_batches > 0:
            avg_loss = epoch_loss / total_batches
            print(f"Epoch [{epoch+1}/{epochs}] Average Weighted Loss: {avg_loss:.6f}")
        
        # 儲存模型 (獨立存放在 v4_student 資料夾，避免覆蓋舊檔案)
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
            
    print("\nTraining Completed! Model weights saved in checkpoints_v4_student/")

if __name__ == '__main__':
    main()