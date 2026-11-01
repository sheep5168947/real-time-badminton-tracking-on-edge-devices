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
    print(f"[INFO] Using device: {device}", flush=True)

    model = MiniROIUNet().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints_v4_student/epoch_30.pth", map_location=device, weights_only=False))
    model.eval()

    csv_list = sorted(list((PROJECT_ROOT / "dataset/raw/part1").glob("match*/csv/*_ball.csv")))
    print(f"[INFO] Checking {len(csv_list)} videos...", flush=True)

    found_cases = []

    for c_path in csv_list:
        stem = c_path.stem.replace("_ball", "")
        v_path = c_path.parent.parent / "video" / f"{stem}.mp4"
        if not v_path.exists():
            continue

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

        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(f, (512, 288)))

            if len(frames) == 3:
                curr_fid = fid - 1
                gt = gt_dict.get(curr_fid, {"vis": 0, "x": 0, "y": 0})

                if gt["vis"] == 1:
                    gx, gy = gt["x"] * sx, gt["y"] * sy

                    # Frame diff mask between frame 1 and frame 0
                    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                    _, m1 = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                    m1_f = m1.astype(np.float32) / 255.0

                    x_in = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                    tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        raw_hm = model(tx).squeeze().cpu().numpy()

                    masked_hm = raw_hm * m1_f

                    rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                    mx, my, mc = rt.get_shuttle_center_local(masked_hm)

                    dist_raw = np.hypot(rx - gx, ry - gy)
                    dist_masked = np.hypot(mx - gx, my - gy)
                    shift = np.hypot(rx - mx, ry - my)

                    # Criteria:
                    # 1. Raw peak is FAR away from GT (> 35 pixels!) -> definitely WRONG / MISSED
                    # 2. Masked peak is CLOSE to GT (<= 6 pixels) -> CORRECT HIT!
                    # 3. Masked conf >= 0.55 -> PASSES THRESHOLD!
                    if dist_raw > 35.0 and dist_masked <= 6.0 and mc >= 0.55:
                        res = {
                            "clip": stem, "fid": curr_fid,
                            "rc": float(rc), "mc": float(mc),
                            "rx": int(rx), "ry": int(ry),
                            "mx": int(mx), "my": int(my),
                            "gx": float(gx), "gy": float(gy),
                            "dist_raw": float(dist_raw),
                            "dist_masked": float(dist_masked),
                            "shift": float(shift)
                        }
                        found_cases.append(res)
                        print(f"★★★ [GENUINE CASE 1 FOUND] {stem} Frame {curr_fid}: "
                              f"Raw=({rx},{ry}) conf={rc:.3f} err={dist_raw:.1f}px -> "
                              f"Masked=({mx},{my}) conf={mc:.3f} err={dist_masked:.1f}px (shift={shift:.1f}px, GT=({gx:.1f},{gy:.1f}))",
                              flush=True)
                        if len(found_cases) >= 5:
                            break

                frames.pop(0)
            fid += 1

        cap.release()
        if len(found_cases) >= 5:
            break

    print(f"\n[SUMMARY] Found {len(found_cases)} genuine cases.", flush=True)

if __name__ == "__main__":
    main()
