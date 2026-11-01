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

    # Search in match1, match2, match3, match4, match5, match9, match10
    v_paths = sorted(list((PROJECT_ROOT / "dataset/raw/part1").glob("match*/video/*.mp4")))[:25]
    print(f"[INFO] Scanning {len(v_paths)} videos for strict Case 1 and Case 2...", flush=True)

    c1_results = []
    c2_results = []

    for v_path in v_paths:
        c_path = v_path.parent.parent / "csv" / f"{v_path.stem}_ball.csv"
        if not c_path.exists():
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
                gt_512 = (gt["x"] * scale_x, gt["y"] * scale_y) if gt["vis"] == 1 else None

                # 1. Global 512 Test
                g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                m_f = m.astype(np.float32) / 255.0

                x_in = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    raw_hm = model(tx).squeeze().cpu().numpy()

                masked_hm = raw_hm * cv2.resize(m_f, (512, 288))
                rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                mx, my, mc = rt.get_shuttle_center_local(masked_hm)
                shift = np.hypot(rx - mx, ry - my)

                # Check Global Case 1
                if gt_512 is not None:
                    err_r = np.hypot(rx - gt_512[0], ry - gt_512[1])
                    err_m = np.hypot(mx - gt_512[0], my - gt_512[1])
                    if err_r > 15.0 and err_m <= 6.0 and mc >= 0.55 and shift >= 12.0:
                        item = {
                            "type": "C1_GLOBAL", "clip": v_path.stem, "fid": curr_fid,
                            "rc": float(rc), "mc": float(mc),
                            "rx": int(rx), "ry": int(ry), "mx": int(mx), "my": int(my),
                            "err_r": float(err_r), "err_m": float(err_m), "shift": float(shift),
                            "frame": frames[1], "mask": m_f, "raw_hm": raw_hm, "masked_hm": masked_hm,
                            "gt_512": gt_512
                        }
                        c1_results.append(item)
                        print(f"★ [GLOBAL C1 FOUND!] {v_path.stem} f{curr_fid}: raw=({rx},{ry})@{rc:.2f} err={err_r:.1f}px -> masked=({mx},{my})@{mc:.2f} err={err_m:.1f}px (shift={shift:.1f}px)", flush=True)

                # Check Global Case 2
                if rc >= 0.55:
                    is_false_alarm = False
                    if gt_512 is None:
                        is_false_alarm = True
                    elif np.hypot(rx - gt_512[0], ry - gt_512[1]) > 25.0:
                        is_false_alarm = True

                    if is_false_alarm and mc < 0.25:
                        item = {
                            "type": "C2_GLOBAL", "clip": v_path.stem, "fid": curr_fid,
                            "rc": float(rc), "mc": float(mc),
                            "rx": int(rx), "ry": int(ry), "mx": int(mx), "my": int(my),
                            "frame": frames[1], "mask": m_f, "raw_hm": raw_hm, "masked_hm": masked_hm,
                            "gt_512": gt_512
                        }
                        c2_results.append(item)
                        print(f"★ [GLOBAL C2 FOUND!] {v_path.stem} f{curr_fid}: raw=({rx},{ry})@{rc:.2f} >= 0.55 -> masked=({mx},{my})@{mc:.2f} < 0.25 (Blocked!)", flush=True)

                # 2. Local 128 Test (when ball is visible, test offset of 15-20px simulating tracking lag)
                if gt_512 is not None:
                    gx, gy = int(gt_512[0]), int(gt_512[1])
                    for ox, oy in [(-18, 0), (18, 0), (0, -18), (0, 18), (-18, -18), (18, 18)]:
                        cx = int(np.clip(gx + ox, 64, 512 - 64))
                        cy = int(np.clip(gy + oy, 64, 288 - 64))

                        c0, _, _ = rt.crop_and_pad(frames[0], cx, cy, 128)
                        c1, _, _ = rt.crop_and_pad(frames[1], cx, cy, 128)
                        c2, _, _ = rt.crop_and_pad(frames[2], cx, cy, 128)

                        gc0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
                        gc1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
                        _, m_c = cv2.threshold(cv2.absdiff(gc1, gc0), 10, 255, cv2.THRESH_BINARY)
                        m_f_c = m_c.astype(np.float32) / 255.0

                        xin_c = np.concatenate([c0, c1, c2], axis=-1).astype(np.float32) / 255.0
                        tx_c = torch.from_numpy(xin_c.transpose(2, 0, 1)).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            r_c = model(tx_c).squeeze().cpu().numpy()

                        m_c_hm = r_c * cv2.resize(m_f_c, (128, 128))
                        lrx, lry, lrc = rt.get_shuttle_center_local(r_c)
                        lmx, lmy, lmc = rt.get_shuttle_center_local(m_c_hm)

                        target_x = 64 - (cx - gx)
                        target_y = 64 - (cy - gy)

                        l_err_r = np.hypot(lrx - target_x, lry - target_y)
                        l_err_m = np.hypot(lmx - target_x, lmy - target_y)
                        l_shift = np.hypot(lrx - lmx, lry - lmy)

                        if l_err_r > 15.0 and l_err_m <= 6.0 and lmc >= 0.55 and l_shift >= 12.0:
                            item = {
                                "type": "C1_LOCAL", "clip": v_path.stem, "fid": curr_fid,
                                "rc": float(lrc), "mc": float(lmc),
                                "rx": int(lrx), "ry": int(lry), "mx": int(lmx), "my": int(lmy),
                                "err_r": float(l_err_r), "err_m": float(l_err_m), "shift": float(l_shift),
                                "frame": c1, "mask": m_f_c, "raw_hm": r_c, "masked_hm": m_c_hm,
                                "crop_center": (cx, cy), "target_pt": (target_x, target_y)
                            }
                            c1_results.append(item)
                            print(f"★ [LOCAL C1 FOUND!] {v_path.stem} f{curr_fid} ({ox},{oy}): raw=({lrx},{lry})@{lrc:.2f} err={l_err_r:.1f}px -> masked=({lmx},{lmy})@{lmc:.2f} err={l_err_m:.1f}px (shift={l_shift:.1f}px)", flush=True)

                        if lrc >= 0.55 and l_err_r > 25.0 and lmc < 0.20:
                            item = {
                                "type": "C2_LOCAL", "clip": v_path.stem, "fid": curr_fid,
                                "rc": float(lrc), "mc": float(lmc),
                                "rx": int(lrx), "ry": int(lry), "mx": int(lmx), "my": int(lmy),
                                "frame": c1, "mask": m_f_c, "raw_hm": r_c, "masked_hm": m_c_hm,
                                "crop_center": (cx, cy), "target_pt": (target_x, target_y)
                            }
                            c2_results.append(item)
                            print(f"★ [LOCAL C2 FOUND!] {v_path.stem} f{curr_fid} ({ox},{oy}): raw=({lrx},{lry})@{lrc:.2f} >= 0.55 -> masked=({lmx},{lmy})@{lmc:.2f} < 0.20 (Blocked!)", flush=True)

                frames.pop(0)
            fid += 1
        cap.release()

        if len(c1_results) >= 3 and len(c2_results) >= 3:
            print("[INFO] Reached 3+ candidates for both cases. Stopping.", flush=True)
            break

    print(f"\n================ SUMMARY ================", flush=True)
    print(f"Case 1 candidates: {len(c1_results)}", flush=True)
    print(f"Case 2 candidates: {len(c2_results)}", flush=True)

    with open(PROJECT_ROOT / "report_materials" / "golden_strict_found.pkl", "wb") as f:
        pickle.dump({"c1": c1_results, "c2": c2_results}, f)
    print("Saved to golden_strict_found.pkl", flush=True)

if __name__ == "__main__":
    main()
