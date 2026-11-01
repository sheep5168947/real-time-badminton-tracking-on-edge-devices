import sys
import os
import glob
import csv
import cv2
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Thesis\tracknet-series-pytorch")
sys.path.insert(0, str(PROJECT_ROOT))
from thesis.models.miniroi_unet import MiniROIUNet
from thesis.platforms.pc import student_realtime_events as rt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints_v4_student/epoch_30.pth", map_location=device, weights_only=False))
    model.eval()

    # Collect all video and CSV pairs
    video_csv_pairs = []
    for csv_file in (PROJECT_ROOT / "dataset/raw/part1").glob("match*/csv/*_ball.csv"):
        v_file = csv_file.parent.parent / "video" / f"{csv_file.stem.replace('_ball', '')}.mp4"
        if v_file.exists():
            video_csv_pairs.append((v_file, csv_file))

    for csv_file in (PROJECT_ROOT / "dataset/raw/am_singles").glob("**/ball_trajectory/*_ball.csv"):
        v_file = csv_file.parent.parent / "rally_video" / f"{csv_file.stem.replace('_ball', '')}.mp4"
        if v_file.exists():
            video_csv_pairs.append((v_file, csv_file))

    print(f"[INFO] Found {len(video_csv_pairs)} video-CSV pairs across Part 1 & Amateur.", flush=True)

    c1_golden = []
    c2_golden = []

    for v_idx, (v_path, c_path) in enumerate(video_csv_pairs):
        gt_dict = {}
        with open(c_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fk = "Frame" if "Frame" in r else "frame_num"
                vk = "Visibility" if "Visibility" in r else "visible"
                xk = "X" if "X" in r else "x"
                yk = "Y" if "Y" in r else "y"
                gt_dict[int(r[fk])] = {"vis": int(r[vk]), "x": float(r[xk]), "y": float(r[yk])}

        cap = cv2.VideoCapture(str(v_path))
        w_orig = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280
        h_orig = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
        scale_x = 512.0 / w_orig
        scale_y = 288.0 / h_orig

        frames = []
        fid = 0

        # We will batch candidate crops
        crops_data = [] # stores (fid, tag, c_images, m_f, target_pt, gt_pt)

        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(f, (512, 288)))

            if len(frames) == 3:
                curr_fid = fid - 1
                gt = gt_dict.get(curr_fid, {"vis": 0, "x": 0, "y": 0})
                gt_pt = (gt["x"] * scale_x, gt["y"] * scale_y) if gt["vis"] == 1 else None

                # Generate Local 128 crops around GT or KF predicted position
                if gt_pt is not None:
                    gx, gy = int(gt_pt[0]), int(gt_pt[1])
                    # Try crop centered at ball and offset by 15-20px (simulating tracking jitter)
                    for ox, oy in [(0, 0), (16, 0), (-16, 0), (0, 16), (0, -16)]:
                        cx = int(np.clip(gx + ox, 64, 512 - 64))
                        cy = int(np.clip(gy + oy, 64, 288 - 64))

                        c0, _, _ = rt.crop_and_pad(frames[0], cx, cy, 128)
                        c1, _, _ = rt.crop_and_pad(frames[1], cx, cy, 128)
                        c2, _, _ = rt.crop_and_pad(frames[2], cx, cy, 128)

                        gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                        gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                        _, m_c = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                        m_f = m_c.astype(np.float32) / 255.0

                        target_pt = (64 - (cx - gx), 64 - (cy - gy))
                        crops_data.append((curr_fid, f"LOCAL_{ox}_{oy}", [c0, c1, c2], m_f, target_pt, gt_pt, (cx, cy)))

                # Also test background crops when ball is far away or invisible (for Case 2)
                if gt["vis"] == 0 or (gt_pt is not None and (gt_pt[0] > 300 or gt_pt[1] < 100)):
                    # Test near bottom player feet / baseline (typical false alarm hotspot)
                    test_positions = [(256, 210), (180, 220), (330, 220), (256, 144)]
                    for t_pos in test_positions:
                        if gt_pt is not None and np.hypot(t_pos[0] - gt_pt[0], t_pos[1] - gt_pt[1]) < 80:
                            continue
                        c0, _, _ = rt.crop_and_pad(frames[0], t_pos[0], t_pos[1], 128)
                        c1, _, _ = rt.crop_and_pad(frames[1], t_pos[0], t_pos[1], 128)
                        c2, _, _ = rt.crop_and_pad(frames[2], t_pos[0], t_pos[1], 128)

                        gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                        gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                        _, m_c = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                        m_f = m_c.astype(np.float32) / 255.0

                        crops_data.append((curr_fid, f"BG_{t_pos[0]}_{t_pos[1]}", [c0, c1, c2], m_f, None, gt_pt, t_pos))

                frames.pop(0)
            fid += 1
        cap.release()

        # Batch inference on crops_data
        if not crops_data:
            continue

        batch_size = 64
        for b_start in range(0, len(crops_data), batch_size):
            batch = crops_data[b_start : b_start + batch_size]
            tensors = []
            for item in batch:
                imgs = np.concatenate(item[2], axis=-1).astype(np.float32) / 255.0
                tensors.append(imgs.transpose(2, 0, 1))
            batch_tx = torch.from_numpy(np.stack(tensors)).to(device)

            with torch.inference_mode():
                batch_raw = model(batch_tx).squeeze(1).cpu().numpy()

            for i, item in enumerate(batch):
                curr_fid, tag, c_imgs, m_f, target_pt, gt_pt, crop_center = item
                raw_hm = batch_raw[i]
                mask_resized = cv2.resize(m_f, (raw_hm.shape[1], raw_hm.shape[0]))
                masked_hm = raw_hm * mask_resized

                rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                mx, my, mc = rt.get_shuttle_center_local(masked_hm)
                shift = np.hypot(rx - mx, ry - my)

                # Check Case 1: Peak Rescued
                if target_pt is not None:
                    err_raw = np.hypot(rx - target_pt[0], ry - target_pt[1])
                    err_masked = np.hypot(mx - target_pt[0], my - target_pt[1])
                    # Strict criteria:
                    # Raw was wrong (>12px err), Masked is on ball (<=6px err),
                    # Masked confidence >= 0.55 (valid hit!), and shift >= 12px
                    if err_raw > 12.0 and err_masked <= 6.0 and mc >= 0.55 and shift >= 12.0:
                        c1_golden.append({
                            "v_path": str(v_path), "clip": v_path.stem, "fid": curr_fid,
                            "tag": tag, "crop_center": crop_center,
                            "rc": float(rc), "mc": float(mc),
                            "rx": int(rx), "ry": int(ry), "mx": int(mx), "my": int(my),
                            "err_raw": float(err_raw), "err_masked": float(err_masked), "shift": float(shift),
                            "crop_img": c_imgs[1], "mask": m_f, "raw_hm": raw_hm, "masked_hm": masked_hm,
                            "target_pt": target_pt, "gt_pt": gt_pt
                        })
                        print(f"★ [FOUND CASE 1] {v_path.stem} f{curr_fid} ({tag}): raw={rc:.3f} (err={err_raw:.1f}px) -> masked={mc:.3f} (err={err_masked:.1f}px, shift={shift:.1f}px)", flush=True)

                # Check Case 2: False Alarm Blocked
                # Raw was false alarm >= 0.55 on static background, and after mask, peak is crushed < 0.20!
                if rc >= 0.55:
                    is_noise = False
                    if target_pt is None:
                        is_noise = True
                    elif np.hypot(rx - target_pt[0], ry - target_pt[1]) > 25.0:
                        is_noise = True

                    if is_noise and mc < 0.20:
                        c2_golden.append({
                            "v_path": str(v_path), "clip": v_path.stem, "fid": curr_fid,
                            "tag": tag, "crop_center": crop_center,
                            "rc": float(rc), "mc": float(mc),
                            "rx": int(rx), "ry": int(ry), "mx": int(mx), "my": int(my),
                            "crop_img": c_imgs[1], "mask": m_f, "raw_hm": raw_hm, "masked_hm": masked_hm,
                            "target_pt": target_pt, "gt_pt": gt_pt
                        })
                        print(f"★ [FOUND CASE 2] {v_path.stem} f{curr_fid} ({tag}): raw={rc:.3f} >= 0.55 -> masked={mc:.3f} < 0.20 (Crushed!)", flush=True)

        if len(c1_golden) >= 5 and len(c2_golden) >= 5:
            print(f"[INFO] Gathered 5+ candidates for both cases at video {v_idx+1}! Finishing scan.", flush=True)
            break

    print(f"\n================ SUMMARY ================", flush=True)
    print(f"Case 1 (Peak Rescued) Total: {len(c1_golden)}", flush=True)
    print(f"Case 2 (False Alarm Blocked) Total: {len(c2_golden)}", flush=True)

    import pickle
    save_path = PROJECT_ROOT / "report_materials" / "golden_strict_cases.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"c1": c1_golden, "c2": c2_golden}, f)
    print(f"[INFO] Successfully saved to {save_path}", flush=True)

if __name__ == "__main__":
    main()
