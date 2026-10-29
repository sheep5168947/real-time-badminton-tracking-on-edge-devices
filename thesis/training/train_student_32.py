
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from thesis.models.miniroi_unet import MiniROIUNet


# === 加權均方誤差損失函數 ===
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
        return (base_loss * weight_mask).mean()


# === 極限 32x32 動態裁切資料集 ===
class CachedVideoDataset32x32(Dataset):
    def __init__(self, pt_file):
        self.data = torch.load(pt_file, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inp, tgt = item["input"], item["target"]

        # 1. 在原 128x128 範圍內尋找目標中心
        max_val = torch.max(tgt)
        if max_val > 0:
            flat_idx = torch.argmax(tgt.reshape(-1))
            by, bx = divmod(flat_idx.item(), 128)
        else:
            bx, by = 64, 64

        # 2. 32x32 半徑僅 16 像素，Jitter 縮至 -5 到 5，防止球體被截斷
        cx = bx + random.randint(-5, 5)
        cy = by + random.randint(-5, 5)

        # 3. 計算 32x32 裁切範圍與安全邊界
        x1, x2 = cx - 16, cx + 16
        y1, y2 = cy - 16, cy + 16

        src_x1, src_x2 = max(0, x1), min(128, x2)
        src_y1, src_y2 = max(0, y1), min(128, y2)

        dst_x1 = max(0, -x1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y1 = max(0, -y1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        new_inp = torch.zeros((9, 32, 32), dtype=inp.dtype)
        new_tgt = torch.zeros((1, 32, 32), dtype=tgt.dtype)

        new_inp[:, dst_y1:dst_y2, dst_x1:dst_x2] = inp[:, src_y1:src_y2, src_x1:src_x2]
        new_tgt[:, dst_y1:dst_y2, dst_x1:dst_x2] = tgt[:, src_y1:src_y2, src_x1:src_x2]

        return new_inp, new_tgt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device}")

    # MiniROIUNet 為全卷積架構，輸入尺寸縮小為 32x32 仍能完成 2 次方整除下採樣
    model = MiniROIUNet().to(device)
    print("Starting training from scratch for 32x32 extreme compression.")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedMSELoss()

    video_pt_files = glob.glob("dataset/cached_pt_v4/*.pt")
    if not video_pt_files:
        print("Error: No .pt files found in dataset/cached_pt_v4/")
        return

    epochs = 30
    batch_size = 16
    save_dir = "checkpoints_v4_student_32x32"
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Starting 32x32 Extreme Compression Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for pt_file in video_pt_files:
            video_name = os.path.basename(pt_file)
            dataset = CachedVideoDataset32x32(pt_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [{video_name}]",
                leave=False,
            )
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1
                pbar.set_postfix({"w_Loss": f"{loss.item():.5f}"})

        if total_batches > 0:
            avg_loss = epoch_loss / total_batches
            print(f"Epoch [{epoch + 1}/{epochs}] Average Weighted Loss: {avg_loss:.6f}")

        torch.save(
            model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
        )

    print(f"\nTraining Completed! Model weights saved in {save_dir}")


if __name__ == "__main__":
    main()