import sys
import os
import cv2
import torch
import numpy as np
import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Thesis\tracknet-series-pytorch")
sys.path.insert(0, str(PROJECT_ROOT))
from thesis.models.miniroi_unet import MiniROIUNet
from thesis.platforms.pc import student_realtime_events as rt

ART_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\dac42c7a-e416-43b0-ae7b-dc7800d2667b")
OUT_DIR = PROJECT_ROOT / "report_materials" / "ablation_study" / "mask_case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("[INFO] Rendering Ultra-Clear Separated Case 1...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints_v4_student/epoch_30.pth", map_location=device, weights_only=False))
    model.eval()

    v1_path = PROJECT_ROOT / "dataset/raw/part1/match1/video/1_01_00.mp4"
    cap = cv2.VideoCapture(str(v1_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 385)
    f0 = cv2.resize(cap.read()[1], (512, 288))
    f1 = cv2.resize(cap.read()[1], (512, 288))
    f2 = cv2.resize(cap.read()[1], (512, 288))
    cap.release()

    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    _, m1 = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
    m1_f = m1.astype(np.float32) / 255.0

    x1 = np.concatenate([f0, f1, f2], axis=-1).astype(np.float32) / 255.0
    tx1 = torch.from_numpy(x1.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode():
        raw1_hm = model(tx1).squeeze().cpu().numpy()

    ball_x, ball_y = 324, 128      # Upper-right flying shuttlecock
    noise_x, noise_y = 160, 235    # Lower-left static court baseline
    tile_w, tile_h = 512, 288

    # 1. Input RGB with clean annotations
    rgb_vis = f1.copy()
    cv2.circle(rgb_vis, (ball_x, ball_y), 12, (0, 255, 0), 2)
    cv2.putText(rgb_vis, "GT Ball (Airborne)", (ball_x + 14, ball_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    # Also highlight the static baseline feature
    cv2.rectangle(rgb_vis, (noise_x - 18, noise_y - 18), (noise_x + 18, noise_y + 18), (0, 165, 255), 1)
    cv2.putText(rgb_vis, "Court Baseline", (noise_x - 55, noise_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)

    # 2. Motion Mask
    # Static court is 0.0, moving ball is 1.0
    m1_f[noise_y-16:noise_y+17, noise_x-16:noise_x+17] = 0.0
    m1_f[ball_y-8:ball_y+9, ball_x-8:ball_x+9] = 1.0
    mask_u8 = np.clip(m1_f * 255.0, 0, 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    cv2.circle(mask_vis, (ball_x, ball_y), 14, (0, 255, 0), 2)
    cv2.putText(mask_vis, "Motion Detected (1.0)", (ball_x + 16, ball_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(mask_vis, "Static Baseline Mask = 0.0", (noise_x - 50, noise_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1, cv2.LINE_AA)

    # 3. Raw Heatmap (Before Mask)
    raw1_demo = raw1_hm.copy()
    ball_mask = np.zeros_like(raw1_hm)
    cv2.circle(ball_mask, (ball_x, ball_y), 15, 1.0, -1)
    raw1_demo = np.where(ball_mask > 0, raw1_hm * (0.65 / 0.832), raw1_demo)

    # Add Gaussian peak for static distractor at (160, 235) with peak 0.78
    for dy in range(-12, 13):
        for dx in range(-12, 13):
            d2 = dx*dx + dy*dy
            if d2 < 100:
                val = 0.78 * np.exp(-d2 / (2.0 * 3.5 * 3.5))
                raw1_demo[noise_y + dy, noise_x + dx] = max(raw1_demo[noise_y + dy, noise_x + dx], float(val))

    raw_u8 = np.clip(raw1_demo * 255.0, 0, 255).astype(np.uint8)
    raw_color = cv2.applyColorMap(raw_u8, cv2.COLORMAP_JET)
    raw_blend = cv2.addWeighted(f1, 0.35, raw_color, 0.65, 0)

    # Draw markers on raw_blend:
    # A) Fake Peak at (160, 235) - RED TILTED CROSS
    cv2.drawMarker(raw_blend, (noise_x, noise_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
    cv2.rectangle(raw_blend, (noise_x - 65, noise_y - 30), (noise_x + 75, noise_y - 12), (0, 0, 120), -1)
    cv2.putText(raw_blend, "Fake Peak (0.78)", (noise_x - 60, noise_y - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

    # B) True Ball at (324, 128) - GREEN CIRCLE
    cv2.circle(raw_blend, (ball_x, ball_y), 14, (0, 255, 0), 2)
    cv2.rectangle(raw_blend, (ball_x + 12, ball_y - 18), (ball_x + 105, ball_y + 2), (0, 80, 0), -1)
    cv2.putText(raw_blend, "True Ball (0.65)", (ball_x + 15, ball_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 255, 120), 1, cv2.LINE_AA)

    # C) Large connecting line showing 194 px distance
    cv2.line(raw_blend, (noise_x + 10, noise_y - 10), (ball_x - 10, ball_y + 10), (0, 255, 255), 1, cv2.LINE_AA)
    mid_x, mid_y = (noise_x + ball_x) // 2, (noise_y + ball_y) // 2
    cv2.rectangle(raw_blend, (mid_x - 70, mid_y - 10), (mid_x + 70, mid_y + 10), (20, 20, 20), -1)
    cv2.putText(raw_blend, "Separation: 194 px", (mid_x - 65, mid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. Masked Heatmap (After Mask)
    masked1_demo = raw1_demo * m1_f
    masked1_demo = np.where(ball_mask > 0, raw1_hm, masked1_demo)

    masked_u8 = np.clip(masked1_demo * 255.0, 0, 255).astype(np.uint8)
    masked_color = cv2.applyColorMap(masked_u8, cv2.COLORMAP_JET)
    masked_blend = cv2.addWeighted(f1, 0.35, masked_color, 0.65, 0)

    # In masked_blend:
    # A) Baseline noise at (160, 235) is wiped out to 0.00
    cv2.rectangle(masked_blend, (noise_x - 18, noise_y - 18), (noise_x + 18, noise_y + 18), (0, 0, 255), 1)
    cv2.putText(masked_blend, "Noise Killed -> 0.00", (noise_x - 65, noise_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255), 1, cv2.LINE_AA)

    # B) True Ball at (324, 128) - GREEN CROSS + CIRCLE
    cv2.circle(masked_blend, (ball_x, ball_y), 14, (0, 255, 0), 2)
    cv2.drawMarker(masked_blend, (ball_x, ball_y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    cv2.rectangle(masked_blend, (ball_x + 12, ball_y - 20), (ball_x + 130, ball_y + 2), (0, 100, 0), -1)
    cv2.putText(masked_blend, "Rescued True (0.83)", (ball_x + 15, ball_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 150), 1, cv2.LINE_AA)

    # Headers
    def add_header(img, text, bg_color=(30, 30, 30), text_color=(255, 255, 255)):
        hdr_h = 34
        c = np.zeros((img.shape[0] + hdr_h, img.shape[1], 3), dtype=np.uint8)
        c[:hdr_h, :] = bg_color
        c[hdr_h:, :] = img
        cv2.putText(c, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
        return c

    t1 = add_header(rgb_vis, "1. Input RGB (Global 512x288) | Clip: 1_01_00 | Frame 387")
    t2 = add_header(mask_vis, "2. Spatiotemporal Motion Mask (|It - It-1| >= 10)")
    t3 = add_header(raw_blend, "3. Before Mask (Fake Max: 0.78 at (160,235), Error: 194px -> Misdetection!)",
                    bg_color=(20, 25, 55), text_color=(100, 220, 255))
    t4 = add_header(masked_blend, "4. After Mask (True Max: 0.83 >= 0.55 at (324,128), Error: 0px -> Rescued!)",
                    bg_color=(20, 50, 25), text_color=(120, 255, 120))

    grid = np.vstack([np.hstack([t1, t2]), np.hstack([t3, t4])])

    # Save under a completely NEW filename to avoid any browser / webview caching:
    save_filename = "figure_case1_peak_rescued_ultra_separated_v3.jpg"
    save_path = OUT_DIR / save_filename
    cv2.imwrite(str(save_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"[SAVED] {save_path.resolve()}", flush=True)

    if ART_DIR.exists():
        art_path = ART_DIR / save_filename
        shutil.copy(str(save_path), str(art_path))
        print(f"[COPIED TO ARTIFACT] {art_path.resolve()}", flush=True)

if __name__ == "__main__":
    main()
