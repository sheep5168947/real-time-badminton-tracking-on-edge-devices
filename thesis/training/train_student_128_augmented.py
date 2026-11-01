# -*- coding: utf-8 -*-
"""
Fine-tuning MiniROIUNet with Online Spatial Jittering Data Augmentation.
Solves the "Data-Shift" problem where training crops were always centered at (64, 64)
while inference crops have off-center shuttlecocks due to physical frame-to-frame displacement.
"""

if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import os
import glob
from tqdm import tqdm

from thesis.models.miniroi_unet import MiniROIUNet

class WeightedMSELoss(nn.Module):
    def __init__(self, ball_weight=10.0, false_positive_penalty=5.0):
        super(WeightedMSELoss, self).__init__()
        self.ball_weight = ball_weight
        self.false_positive_penalty = false_positive_penalty

    def forward(self, pred, target):
        base_loss = (pred - target) ** 2
        weight_mask = torch.ones_like(target)
        weight_mask[target > 0] = self.ball_weight
        false_positive_mask = (target == 0) & (pred > 0.1)
        weight_mask[false_positive_mask] = self.false_positive_penalty
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

class AugmentedCachedVideoDataset(Dataset):
    def __init__(self, pt_file, max_jitter=28, p_jitter=0.75):
        self.data = torch.load(pt_file, weights_only=False)
        self.max_jitter = max_jitter
        self.p_jitter = p_jitter
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        inp, tgt = item["input"], item["target"] # (9, 128, 128), (1, 128, 128)
        
        # 75% probability of applying random translation
        if np.random.rand() < self.p_jitter:
            dx = int(np.random.randint(-self.max_jitter, self.max_jitter + 1))
            dy = int(np.random.randint(-self.max_jitter, self.max_jitter + 1))
            inp = TF.affine(inp, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            tgt = TF.affine(tgt, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            
        return inp, tgt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN] Fine-tuning on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    model = MiniROIUNet().to(device)
    base_weights = r"checkpoints_v4_student\epoch_30.pth"
    
    if os.path.exists(base_weights):
        print(f"[TRAIN] Loading base weights from {base_weights} for fine-tuning...")
        ckpt = torch.load(base_weights, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("[TRAIN] Warning: Base weights not found, initializing from scratch.")

    # Slightly lower lr for fine-tuning spatial shift tolerance
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = WeightedMSELoss(ball_weight=10.0, false_positive_penalty=5.0)
    
    video_pt_files = glob.glob(r"dataset\cached_pt_v4\*.pt")
    if not video_pt_files:
        print("[ERROR] No .pt files found in dataset/cached_pt_v4/")
        return
        
    epochs = 15
    batch_size = 32
    save_dir = "checkpoints_v4_student_augmented"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[TRAIN] Found {len(video_pt_files)} video caches. Total epochs: {epochs}, Batch size: {batch_size}")
    print("[TRAIN] Starting Spatial Jittering Fine-Tuning...")
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        
        # Shuffle order of videos each epoch
        np.random.shuffle(video_pt_files)
        
        for pt_file in video_pt_files:
            dataset = AugmentedCachedVideoDataset(pt_file, max_jitter=28, p_jitter=0.75)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_batches += 1
                
        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
        print(f"Epoch [{epoch+1:02d}/{epochs}] Average Weighted Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_augmented.pth"))
            print(f"  --> Saved new best augmented model with loss: {best_loss:.6f}")
            
    print(f"\n[SUCCESS] Training Completed! Checkpoints saved in {save_dir}/")

if __name__ == '__main__':
    main()
