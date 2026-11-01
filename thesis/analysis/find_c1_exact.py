import sys
import os
import glob
import csv
import cv2
import torch
import numpy as np
import pickle
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
    print(f"[INFO] Checking {len(csv_list)} rallies for Golden Case 1...", flush=True)

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
        scale_x = 512.0 / w_orig
        scale_y = 288.0 / h_orig

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
                    gx = int(gt["x"] * scale_x)
                    gy = int(gt["y"] * scale_y)

                    # Offsets to test:
                    # When shuttlecock is at various positions inside the 128x128 crop
                    for ox in [-30, -20, -10, 0, 10, 20, 30]:
                        for oy in [-30, -20, -10, 0, 10, 20, 30]:
                            cx = int(np.clip(gx + ox, 64, 512 - 64))
                            cy = int(np.clip(gy + oy, 64, 288 - 64))

                            c0, _, _ = rt.crop_and_pad(frames[0], cx, cy, 128)
                            c1, _, _ = rt.crop_and_pad(frames[1], cx, cy, 128)
                            c2, _, _ = rt.crop_and_pad(frames[2], cx, cy, 128)

                            gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                            gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                            _, m_c = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                            m_f = m_c.astype(np.float32) / 255.0

                            x_in = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) / 255.0
                            tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                            with torch.inference_mode():
                                raw_hm = model(tx).squeeze().cpu().numpy()

                            masked_hm = raw_hm * cv2.resize(m_f, (128, 128))

                            rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                            mx, my, mc = rt.get_shuttle_center_local(masked_hm)

                            target_x = 64 - (cx - gx)
                            target_y = 64 - (cy - gy)

                            err_r = np.hypot(rx - target_x, ry - target_y)
                            err_m = np.hypot(mx - target_x, my - target_y)
                            shift = np.hypot(rx - mx, ry - my)

                            # Golden Case 1 criteria:
                            # Raw error > 12px, Masked error <= 6px, mc >= 0.55, shift >= 10px
                            if err_r > 12.0 and err_m <= 6.0 and mc >= 0.55 and shift >= 10.0:
                                res = {
                                    "clip": stem, "fid": curr_fid, "offset": (ox, oy), "crop_center": (cx, cy),
                                    "rc": float(rc), "mc": float(mc),
                                    "rx": int(rx), "ry": int(ry), "mx": int(mx), "my": int(my),
                                    "err_r": float(err_r), "err_m": float(err_m), "shift": float(shift),
                                    "target_pt": (float(target_x), float(target_y)),
                                    "c1": c1, "m_f": m_f, "raw_hm": raw_hm, "masked_hm": masked_hm,
                                    "v_path": str(v_path)
                                }
                                found_cases.append(res)
                                print(f"★★★ [PERFECT CASE 1 FOUND!] {stem} fid={curr_fid} offset=({ox},{oy}): "
                                      f"Raw=({rx},{ry}) conf={rc:.3f} err={err_r:.1f}px -> "
                                      f"Masked=({mx},{my}) conf={mc:.3f} err={err_m:.1f}px (shift={shift:.1f}px)", flush=True)
                                if len(found_cases) >= 5:
                                    break
                        if len(found_cases) >= 5:
                            break
                frames.pop(0)
            fid += 1
        cap.release()
        if len(found_cases) >= 5:
            break

    print(f"\nTotal Case 1 Found: {len(found_cases)}", flush=True)
    if found_cases:
        with open(PROJECT_ROOT / "report_materials" / "c1_perfect.pkl", "wb") as f:
            pickle.dump(found_cases, f)
        print("Saved to c1_perfect.pkl", flush=True)

if __name__ == "__main__":
    main()
