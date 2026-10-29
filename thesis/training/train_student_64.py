
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
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

        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()


# === 極限 64x64 動態裁切資料集 ===
class CachedVideoDataset64x64(Dataset):
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

        # 2. 加入等比例縮小的 Jitter (-12 to 12)
        cx = bx + random.randint(-12, 12)
        cy = by + random.randint(-12, 12)

        # 3. 計算 64x64 裁切範圍與安全邊界防護
        x1, x2 = cx - 32, cx + 32
        y1, y2 = cy - 32, cy + 32

        src_x1, src_x2 = max(0, x1), min(128, x2)
        src_y1, src_y2 = max(0, y1), min(128, y2)

        dst_x1 = max(0, -x1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y1 = max(0, -y1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        new_inp = torch.zeros((9, 64, 64), dtype=inp.dtype)
        new_tgt = torch.zeros((1, 64, 64), dtype=tgt.dtype)

        new_inp[:, dst_y1:dst_y2, dst_x1:dst_x2] = inp[:, src_y1:src_y2, src_x1:src_x2]
        new_tgt[:, dst_y1:dst_y2, dst_x1:dst_x2] = tgt[:, src_y1:src_y2, src_x1:src_x2]

        return new_inp, new_tgt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device: {}".format(device))

    # 初始化模型 (UNet 通常支援輸入維度為 2 的次方縮放，64x64 可直接送入)
    model = MiniROIUNet().to(device)
    print("Starting training from scratch for 64x64 extreme compression.")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedMSELoss()

    video_pt_files = glob.glob("dataset/cached_pt_v4/*.pt")

    if not video_pt_files:
        print("Error: No .pt files found in dataset/cached_pt_v4/")
        return

    epochs = 30
    batch_size = 16
    save_dir = "checkpoints_v4_student_64x64"
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Starting 64x64 Extreme Compression Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0

        for pt_file in video_pt_files:
            video_name = os.path.basename(pt_file)
            print("\nLoading cache: {} ...".format(video_name))

            # 使用新版 64x64 動態裁切 Dataset
            dataset = CachedVideoDataset64x64(pt_file)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            pbar = tqdm(
                dataloader,
                desc="Epoch {}/{} [{}]".format(epoch + 1, epochs, video_name),
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
                pbar.set_postfix({"w_Loss": "{:.5f}".format(loss.item())})

        if total_batches > 0:
            avg_loss = epoch_loss / total_batches
            print(
                "Epoch [{}/{}] Average Weighted Loss: {:.6f}".format(
                    epoch + 1, epochs, avg_loss
                )
            )

        torch.save(
            model.state_dict(), os.path.join(save_dir, "epoch_{}.pth".format(epoch + 1))
        )

    print("\nTraining Completed! Model weights saved in {}".format(save_dir))


if __name__ == "__main__":
    main()
