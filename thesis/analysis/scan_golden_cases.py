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
    print(f"[INFO] Using device: {device}")

    model = MiniROIUNet().to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints_v4_student" / "epoch_30.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    # Search in part1 and am_singles
    p1_vids = sorted(list(PROJECT_ROOT.glob("dataset/raw/part1/match*/video/*.mp4")))
    am_vids = sorted(list(PROJECT_ROOT.glob("dataset/raw/am_singles/**/*.mp4")))
    all_vids = p1_vids[:25] + am_vids[:10]

    print(f"[INFO] Scanning {len(all_vids)} videos for Golden Cases 1 & 2...")

    case1_rescued = []
    case2_blocked = []

    for v_idx, v_path in enumerate(all_vids):
        # Look for corresponding csv
        csv_path = None
        if "part1" in str(v_path):
            candidate_csv = v_path.parent.parent / "csv" / f"{v_path.stem}_ball.csv"
            if candidate_csv.exists():
                csv_path = candidate_csv
        elif "am_singles" in str(v_path):
            candidate_csv = v_path.parent / f"{v_path.stem}_ball.csv"
            if candidate_csv.exists():
                csv_path = candidate_csv

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
                _, m_mask = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
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
                if gt_info is not None and gt_info["vis"] == 1:
                    gx_gt = gt_info["x"] * (512.0 / 1280.0)
                    gy_gt = gt_info["y"] * (288.0 / 720.0)
                    r_err = np.hypot(rx - gx_gt, ry - gy_gt)
                    m_err = np.hypot(mx - gx_gt, my - gy_gt)
                    shift = np.hypot(rx - mx, ry - my)

                    # Case 1 (Peak Rescued):
                    # Raw was on noise (>12px error), Masked is on ball (<=8px error), and masked_conf >= 0.55!
                    if r_err > 12.0 and m_err <= 8.0 and m_conf >= 0.55:
                        case1_rescued.append({
                            "video_path": str(v_path),
                            "frame_id": curr_fid,
                            "scale": "GLOBAL_512",
                            "raw_conf": float(r_conf),
                            "masked_conf": float(m_conf),
                            "r_err": float(r_err),
                            "m_err": float(m_err),
                            "shift": float(shift),
                            "gt": (float(gx_gt), float(gy_gt)),
                            "raw_peak": (float(rx), float(ry)),
                            "masked_peak": (float(mx), float(my))
                        })

                # Case 2 (False Alarm Blocked):
                # Raw peak was a false alarm >= 0.55 on static background (or no ball),
                # and after mask, ALL peaks are < 0.35!
                if r_conf >= 0.55:
                    is_noise = False
                    if gt_info is None or gt_info["vis"] == 0:
                        is_noise = True
                    else:
                        gx_gt = gt_info["x"] * (512.0 / 1280.0)
                        gy_gt = gt_info["y"] * (288.0 / 720.0)
                        if np.hypot(rx - gx_gt, ry - gy_gt) > 25.0:
                            is_noise = True

                    if is_noise and m_conf < 0.35:
                        case2_blocked.append({
                            "video_path": str(v_path),
                            "frame_id": curr_fid,
                            "scale": "GLOBAL_512",
                            "raw_conf": float(r_conf),
                            "masked_conf": float(m_conf),
                            "raw_peak": (float(rx), float(ry)),
                            "masked_peak": (float(mx), float(my))
                        })

                # 2. Local 128 simulation around GT or around noise
                if gt_info is not None and gt_info["vis"] == 1:
                    # Let's also test local crop centered with small jitter (e.g. KF prediction error 15px)
                    for offset in [(0, 0), (12, -8), (-10, 14)]:
                        cx = int(np.clip(gt_info["x"] * (512.0 / 1280.0) + offset[0], 64, 512 - 64))
                        cy = int(np.clip(gt_info["y"] * (288.0 / 720.0) + offset[1], 64, 288 - 64))

                        c0, _, _ = crop_and_pad(f0, cx, cy, 128)
                        c1, _, _ = crop_and_pad(f1, cx, cy, 128)
                        c2, _, _ = crop_and_pad(f2, cx, cy, 128)

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

                        local_gt_x = 64 - offset[0]
                        local_gt_y = 64 - offset[1]

                        lr_err = np.hypot(lrx - local_gt_x, lry - local_gt_y)
                        lm_err = np.hypot(lmx - local_gt_x, lmy - local_gt_y)

                        if lr_err > 12.0 and lm_err <= 6.0 and lm_conf >= 0.55:
                            case1_rescued.append({
                                "video_path": str(v_path),
                                "frame_id": curr_fid,
                                "scale": "LOCAL_128",
                                "raw_conf": float(lr_conf),
                                "masked_conf": float(lm_conf),
                                "r_err": float(lr_err),
                                "m_err": float(lm_err),
                                "shift": float(np.hypot(lrx - lmx, lry - lmy)),
                                "gt": (float(local_gt_x), float(local_gt_y)),
                                "crop_center": (cx, cy)
                            })

                        if lr_conf >= 0.55 and lr_err > 20.0 and lm_conf < 0.35:
                            case2_blocked.append({
                                "video_path": str(v_path),
                                "frame_id": curr_fid,
                                "scale": "LOCAL_128",
                                "raw_conf": float(lr_conf),
                                "masked_conf": float(lm_conf),
                                "crop_center": (cx, cy)
                            })

                frames.pop(0)
            fid += 1
        cap.release()
        if (v_idx + 1) % 5 == 0:
            print(f"[PROGRESS] Checked {v_idx + 1}/{len(all_vids)} videos. Case 1: {len(case1_rescued)}, Case 2: {len(case2_blocked)}")

    print(f"\n================ SUMMARY ================")
    print(f"Total Case 1 (Peak Rescued) found: {len(case1_rescued)}")
    for i, c in enumerate(case1_rescued[:10]):
        print(f"  [{i+1}] {Path(c['video_path']).name} frame {c['frame_id']} ({c['scale']}): raw={c['raw_conf']:.3f} (err={c['r_err']:.1f}px) -> masked={c['masked_conf']:.3f} (err={c['m_err']:.1f}px, shift={c['shift']:.1f}px)")

    print(f"\nTotal Case 2 (False Alarm Blocked) found: {len(case2_blocked)}")
    for i, c in enumerate(case2_blocked[:10]):
        print(f"  [{i+1}] {Path(c['video_path']).name} frame {c['frame_id']} ({c['scale']}): raw={c['raw_conf']:.3f} -> masked={c['masked_conf']:.3f}")

if __name__ == "__main__":
    main()
