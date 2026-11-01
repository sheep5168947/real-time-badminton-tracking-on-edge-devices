import sys
import os
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
    print(f"[INFO] Device: {device}", flush=True)

    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(
        PROJECT_ROOT / "checkpoints_v4_student/epoch_30.pth",
        map_location=device, weights_only=False))
    model.eval()

    # Let's test on 10 diverse rallies (both part1 and am_singles)
    test_videos = [
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_01_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_02_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_03_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_04_00.mp4",
        PROJECT_ROOT / "dataset/raw/part1/match1/video/1_05_00.mp4",
        PROJECT_ROOT / "dataset/raw/am_singles/am_singles/match_china2/rally_video/singles0.mp4",
        PROJECT_ROOT / "dataset/raw/am_singles/am_singles/match_china2/rally_video/singles1.mp4",
        PROJECT_ROOT / "dataset/raw/am_singles/am_singles/match_yewtee/rally_video/singles0.mp4",
    ]

    print("=" * 80)
    print("EXPERIMENT 1: LOCAL SEARCH 128x128 TRACKING - SCANNING FOR CASE 1 & CASE 2")
    print("=" * 80)

    local_case2_found = [] # Ball out of crop / vis=0, local raw_conf >= 0.55, masked_conf < 0.55
    local_case1_found = [] # Ball in crop, raw picks noise (err > 15px), masked picks ball (err <= 5px), mc >= 0.35
    local_crops_evaluated = 0
    local_out_of_crop_count = 0
    local_noise_exceeded_55_count = 0

    for v_path in test_videos:
        if not v_path.exists():
            continue
        stem = v_path.stem
        # find csv
        c_candidates = list(v_path.parent.parent.glob(f"**/{stem}_ball.csv"))
        if not c_candidates:
            c_candidates = list(v_path.parent.parent.glob(f"**/*{stem}*.csv"))
        if not c_candidates:
            continue
        c_path = c_candidates[0]

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
        sx, sy = 512.0 / w_orig, 288.0 / h_orig

        frames = []
        fid = 0
        # Simulated tracker state:
        last_x, last_y = 256, 144
        tracking = True

        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(f, (512, 288)))

            if len(frames) == 3:
                curr_fid = fid - 1
                gt = gt_dict.get(curr_fid, {"vis": 0, "x": 0, "y": 0})
                gx = gt["x"] * sx if gt["vis"] == 1 else -1000
                gy = gt["y"] * sy if gt["vis"] == 1 else -1000

                # Crop around last tracked position (simulate local search 128x128)
                cx, cy = int(last_x), int(last_y)
                c0, _, _ = rt.crop_and_pad(frames[0], cx, cy, 128)
                c1, _, _ = rt.crop_and_pad(frames[1], cx, cy, 128)
                c2, _, _ = rt.crop_and_pad(frames[2], cx, cy, 128)

                gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                _, mc = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                mf = mc.astype(np.float32) / 255.0

                x_in = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) / 255.0
                tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    raw_hm = model(tx).squeeze().cpu().numpy()
                masked_hm = raw_hm * mf

                rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                mx, my, mc = rt.get_shuttle_center_local(masked_hm)

                local_crops_evaluated += 1

                # Is the ball actually inside this 128x128 crop?
                ball_in_crop = (gt["vis"] == 1 and abs(gx - cx) < 60 and abs(gy - cy) < 60)

                if not ball_in_crop:
                    local_out_of_crop_count += 1
                    # Did raw noise exceed 0.55 when ball is out of crop?
                    if rc >= 0.55:
                        local_noise_exceeded_55_count += 1
                        print(f"  [NOISE >= 0.55 in empty crop!] {stem} f{curr_fid}: raw_conf={rc:.3f} at local ({rx},{ry})", flush=True)
                        if mc < 0.55:
                            local_case2_found.append({
                                "clip": stem, "fid": curr_fid, "rc": rc, "mc": mc,
                                "rx": rx, "ry": ry, "crop_c": (cx, cy)
                            })
                            print(f"    --> [LOCAL CASE 2 FOUND!] Raw={rc:.3f} >= 0.55 -> Masked={mc:.3f} < 0.55 (BLOCKED!)", flush=True)
                else:
                    # Ball is in crop
                    local_gx = 64 + (gx - cx)
                    local_gy = 64 + (gy - cy)
                    err_r = np.hypot(rx - local_gx, ry - local_gy)
                    err_m = np.hypot(mx - local_gx, my - local_gy)
                    if err_r > 15.0 and err_m <= 6.0 and mc >= 0.35:
                        local_case1_found.append({
                            "clip": stem, "fid": curr_fid, "rc": rc, "mc": mc,
                            "err_r": err_r, "err_m": err_m
                        })
                        print(f"  [LOCAL CASE 1 FOUND!] {stem} f{curr_fid}: err_r={err_r:.1f} -> err_m={err_m:.1f} rc={rc:.3f} mc={mc:.3f}", flush=True)

                # Update tracking position for next frame
                if gt["vis"] == 1:
                    last_x, last_y = gx, gy
                else:
                    last_x, last_y = 256, 144

                frames.pop(0)
            fid += 1
        cap.release()

    print("\n--- EXPERIMENT 1 SUMMARY (Local 128x128 Tracking) ---")
    print(f"Total 128x128 crops evaluated: {local_crops_evaluated}")
    print(f"Total frames where ball was OUT of crop: {local_out_of_crop_count}")
    print(f"Frames where raw noise >= 0.55 in empty crop: {local_noise_exceeded_55_count}")
    print(f"Local Case 2 (False Alarm Blocked by Mask): {len(local_case2_found)}")
    print(f"Local Case 1 (Peak Rescued by Mask): {len(local_case1_found)}")

    # =========================================================================
    # EXPERIMENT 2: WHAT IF NO CONFIDENCE THRESHOLD IS SET (Pure Argmax)?
    # Compare 4 Modes on 1_01_00:
    # A: Threshold = 0.55, No Mask
    # B: Threshold = 0.55, With Mask
    # C: Threshold = 0.00 (Pure Argmax), No Mask
    # D: Threshold = 0.00 (Pure Argmax), With Mask
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: EFFECT OF CONFIDENCE THRESHOLD (Threshold=0.55 vs Threshold=0.0 / Argmax)")
    print("=" * 80)

    v_bench = PROJECT_ROOT / "dataset/raw/part1/match1/video/1_01_00.mp4"
    c_bench = PROJECT_ROOT / "dataset/raw/part1/match1/csv/1_01_00_ball.csv"
    gt_bench = {}
    with open(c_bench, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt_bench[int(r["Frame"])] = {"vis": int(r["Visibility"]), "x": float(r["X"]), "y": float(r["Y"])}

    for thresh in [0.55, 0.00]:
        for use_mask in [False, True]:
            cap = cv2.VideoCapture(str(v_bench))
            frames = []
            fid = 0
            TP, FP, FN = 0, 0, 0
            t_start = cv2.getTickCount()

            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(f, (512, 288)))

                if len(frames) == 3:
                    curr_fid = fid - 1
                    gt = gt_bench.get(curr_fid, {"vis": 0, "x": 0, "y": 0})
                    gx = gt["x"] * (512.0 / 1280.0)
                    gy = gt["y"] * (288.0 / 720.0)

                    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)

                    x_in = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                    tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        hm = model(tx).squeeze().cpu().numpy()

                    if use_mask:
                        _, m = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                        hm = hm * (m.astype(np.float32) / 255.0)

                    px, py, pconf = rt.get_shuttle_center_local(hm)

                    # Detection decision:
                    has_det = (pconf > thresh) if thresh > 0 else (pconf > 0.0001)

                    if gt["vis"] == 1:
                        if has_det:
                            err = np.hypot(px - gx, py - gy)
                            if err <= 6.0:
                                TP += 1
                            else:
                                FP += 1
                        else:
                            FN += 1
                    else:
                        if has_det:
                            FP += 1

                    frames.pop(0)
                fid += 1
            cap.release()

            t_elapsed = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
            fps = fid / t_elapsed
            prec = (TP / (TP + FP) * 100.0) if (TP + FP) > 0 else 0.0
            rec = (TP / (TP + FN) * 100.0) if (TP + FN) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            tag = f"Thresh={thresh:.2f} | Mask={'ON ' if use_mask else 'OFF'}"
            print(f"[{tag}] TP={TP:3d}, FP={FP:3d}, FN={FN:3d} | Prec={prec:5.2f}%, Rec={rec:5.2f}%, F1={f1:5.2f}% | FPS={fps:.1f}", flush=True)

if __name__ == "__main__":
    main()
