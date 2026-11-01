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
from thesis.platforms.pc.student_realtime_events import get_shuttle_center_local, crop_and_pad

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    model = MiniROIUNet().to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints_v4_student" / "epoch_30.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    # Select representative videos with diverse courts and camera angles
    target_videos = [
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_01_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_01_01.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match2/video/1_02_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match3/video/1_02_01.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match5/video/1_03_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match9/video/1_04_05.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match10/video/1_05_00.mp4",
        PROJECT_ROOT / "dataset/raw/am_singles/am_singles/video/am_01.mp4",
        PROJECT_ROOT / "dataset/raw/am_singles/am_singles/video/am_02.mp4",
    ]

    case1_rescued = []
    case2_blocked = []

    for v_path in target_videos:
        if not v_path.exists():
            continue
        print(f"\n---> Scanning {v_path.stem}...", flush=True)

        # Find CSV
        csv_path = None
        if "part1" in str(v_path):
            c_csv = v_path.parent.parent / "csv" / f"{v_path.stem}_ball.csv"
            if c_csv.exists():
                csv_path = c_csv
        elif "am_singles" in str(v_path):
            c_csv = v_path.parent.parent / "csv" / f"{v_path.stem}_ball.csv"
            if not c_csv.exists():
                c_csv = v_path.parent / f"{v_path.stem}_ball.csv"
            if c_csv.exists():
                csv_path = c_csv

        gt = {}
        if csv_path and csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    f_num = int(row.get("frame_num", row.get("Frame", 0)))
                    vis = int(row.get("visible", row.get("Visibility", 0)))
                    x = float(row.get("x", row.get("X", 0)))
                    y = float(row.get("y", row.get("Y", 0)))
                    gt[f_num] = {"vis": vis, "x": x, "y": y}

        cap = cv2.VideoCapture(str(v_path))
        frames = []
        fid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (512, 288)))

            if len(frames) == 3:
                curr_fid = fid - 1
                f0, f1, f2 = frames[0], frames[1], frames[2]

                g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(g1, g0)
                _, m_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                m_mask = m_mask.astype(np.float32) / 255.0

                # 1. Global 512
                x_in = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    raw_hm = model(tx).squeeze().cpu().numpy()

                mask_resized = cv2.resize(m_mask, (raw_hm.shape[1], raw_hm.shape[0]))
                masked_hm = raw_hm * mask_resized

                rx, ry, r_conf = get_shuttle_center_local(raw_hm)
                mx, my, m_conf = get_shuttle_center_local(masked_hm)

                gt_info = gt.get(curr_fid, None)

                # Check Category 1 on Global
                if gt_info is not None and gt_info["vis"] == 1:
                    gx_gt = gt_info["x"] * (512.0 / 1280.0)
                    gy_gt = gt_info["y"] * (288.0 / 720.0)
                    r_err = np.hypot(rx - gx_gt, ry - gy_gt)
                    m_err = np.hypot(mx - gx_gt, my - gy_gt)
                    shift = np.hypot(rx - mx, ry - my)

                    if r_err > 12.0 and m_err <= 8.0 and m_conf >= 0.55:
                        item = {
                            "video": v_path.stem, "video_path": str(v_path), "fid": curr_fid,
                            "scale": "GLOBAL_512", "raw_conf": float(r_conf), "masked_conf": float(m_conf),
                            "r_err": float(r_err), "m_err": float(m_err), "shift": float(shift),
                            "gt": (float(gx_gt), float(gy_gt)), "raw_peak": (float(rx), float(ry)),
                            "masked_peak": (float(mx), float(my))
                        }
                        case1_rescued.append(item)
                        print(f"  [CASE 1 GLOBAL FOUND!] frame {curr_fid}: raw={r_conf:.2f} (err={r_err:.1f}px) -> masked={m_conf:.2f} (err={m_err:.1f}px)", flush=True)

                # Check Category 2 on Global
                # Raw false alarm >= 0.55, but masked peak is suppressed below 0.35!
                if r_conf >= 0.55:
                    is_bg_noise = False
                    if gt_info is None or gt_info["vis"] == 0:
                        is_bg_noise = True
                    else:
                        gx_gt = gt_info["x"] * (512.0 / 1280.0)
                        gy_gt = gt_info["y"] * (288.0 / 720.0)
                        if np.hypot(rx - gx_gt, ry - gy_gt) > 25.0:
                            is_bg_noise = True

                    if is_bg_noise and m_conf < 0.35:
                        item = {
                            "video": v_path.stem, "video_path": str(v_path), "fid": curr_fid,
                            "scale": "GLOBAL_512", "raw_conf": float(r_conf), "masked_conf": float(m_conf),
                            "raw_peak": (float(rx), float(ry)), "masked_peak": (float(mx), float(my))
                        }
                        case2_blocked.append(item)
                        print(f"  [CASE 2 GLOBAL FOUND!] frame {curr_fid}: raw={r_conf:.2f} -> masked={m_conf:.2f} (Cleanly Blocked!)", flush=True)

                # 2. Local 128 check
                if gt_info is not None and gt_info["vis"] == 1:
                    gx_gt = int(gt_info["x"] * (512.0 / 1280.0))
                    gy_gt = int(gt_info["y"] * (288.0 / 720.0))

                    # Test crop at GT position
                    c0, _, _ = crop_and_pad(f0, gx_gt, gy_gt, 128)
                    c1, _, _ = crop_and_pad(f1, gx_gt, gy_gt, 128)
                    c2, _, _ = crop_and_pad(f2, gx_gt, gy_gt, 128)

                    gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                    gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                    _, mc = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                    mc = mc.astype(np.float32) / 255.0

                    cin = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) / 255.0
                    tc = torch.from_numpy(cin.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        r_c = model(tc).squeeze().cpu().numpy()

                    m_c = r_c * cv2.resize(mc, (r_c.shape[1], r_c.shape[0]))
                    lrx, lry, lr_conf = get_shuttle_center_local(r_c)
                    lmx, lmy, lm_conf = get_shuttle_center_local(m_c)

                    lr_err = np.hypot(lrx - 64, lry - 64)
                    lm_err = np.hypot(lmx - 64, lmy - 64)

                    if lr_err > 12.0 and lm_err <= 6.0 and lm_conf >= 0.55:
                        item = {
                            "video": v_path.stem, "video_path": str(v_path), "fid": curr_fid,
                            "scale": "LOCAL_128", "raw_conf": float(lr_conf), "masked_conf": float(lm_conf),
                            "r_err": float(lr_err), "m_err": float(lm_err), "shift": float(np.hypot(lrx - lmx, lry - lmy)),
                            "gt": (64.0, 64.0), "crop_center": (gx_gt, gy_gt),
                            "raw_peak": (float(lrx), float(lry)), "masked_peak": (float(lmx), float(lmy))
                        }
                        case1_rescued.append(item)
                        print(f"  [CASE 1 LOCAL FOUND!] frame {curr_fid}: raw={lr_conf:.2f} (err={lr_err:.1f}px) -> masked={lm_conf:.2f} (err={lm_err:.1f}px)", flush=True)

                frames.pop(0)
            fid += 1
        cap.release()

    print("\n================ FINAL REPORT ================", flush=True)
    print(f"Case 1 Total: {len(case1_rescued)}", flush=True)
    for c in case1_rescued:
        print("  C1:", c, flush=True)
    print(f"Case 2 Total: {len(case2_blocked)}", flush=True)
    for c in case2_blocked:
        print("  C2:", c, flush=True)

if __name__ == "__main__":
    main()
