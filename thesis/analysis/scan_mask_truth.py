"""
Exhaustive scan: Does Motion Mask help or hurt in practice?

Scan 1: Natural Case 2 (False Alarm Blocked)
  - GT visibility = 0 (no ball), raw_conf >= 0.55, masked_conf < 0.55
  - This would prove mask blocks a false alarm

Scan 2: Motion Mask HURTS (Correct Detection Killed)
  - GT visibility = 1 (ball present), raw correctly finds ball (err <= 5px, conf >= 0.55)
  - After mask, detection is killed (masked_conf < 0.55 OR masked_err > 10px)
  - This would prove mask damages correct detections

Scan 3: Natural Case 1 (Peak Rescued)
  - GT visibility = 1, raw_err > 15px, masked_err <= 6px, masked_conf >= 0.35
"""
import sys, csv, cv2, torch, numpy as np
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

    # Collect all CSVs
    csv_list = sorted(list((PROJECT_ROOT / "dataset/raw/part1").glob("match*/csv/*_ball.csv")))
    am_csvs = list((PROJECT_ROOT / "dataset/raw/am_singles").glob("**/*_ball.csv"))
    csv_list.extend(am_csvs)
    print(f"[INFO] Total CSVs to scan: {len(csv_list)}", flush=True)

    case2_found = []  # False alarm blocked
    mask_hurts = []   # Mask kills correct detection
    case1_found = []  # Peak rescued

    total_visible = 0
    total_invisible = 0
    total_raw_correct = 0
    total_masked_correct = 0

    for ci, c_path in enumerate(csv_list):
        stem = c_path.stem.replace("_ball", "")
        # Try multiple video dirs
        v_path = c_path.parent.parent / "video" / f"{stem}.mp4"
        if not v_path.exists():
            v_path = c_path.parent.parent / "rally_video" / f"{stem}.mp4"
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

                g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(cv2.absdiff(g1, g0), 10, 255, cv2.THRESH_BINARY)
                mf = m.astype(np.float32) / 255.0

                x_in = np.concatenate(frames, axis=-1).astype(np.float32) / 255.0
                tx = torch.from_numpy(x_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    raw_hm = model(tx).squeeze().cpu().numpy()
                masked_hm = raw_hm * mf

                rx, ry, rc = rt.get_shuttle_center_local(raw_hm)
                mx, my, mc = rt.get_shuttle_center_local(masked_hm)

                if gt["vis"] == 1:
                    total_visible += 1
                    gx, gy = gt["x"] * sx, gt["y"] * sy
                    err_raw = np.hypot(rx - gx, ry - gy)
                    err_masked = np.hypot(mx - gx, my - gy)

                    raw_correct = (err_raw <= 5.0 and rc >= 0.55)
                    masked_correct = (err_masked <= 5.0 and mc >= 0.55)

                    if raw_correct:
                        total_raw_correct += 1
                    if masked_correct:
                        total_masked_correct += 1

                    # Case 1: Peak Rescued (raw wrong, masked correct)
                    if err_raw > 15.0 and err_masked <= 6.0 and mc >= 0.35:
                        case1_found.append({
                            "clip": stem, "fid": curr_fid,
                            "rc": rc, "mc": mc,
                            "err_raw": err_raw, "err_masked": err_masked
                        })
                        print(f"  [CASE1 RESCUED] {stem} f{curr_fid}: raw_err={err_raw:.1f} masked_err={err_masked:.1f} rc={rc:.3f} mc={mc:.3f}", flush=True)

                    # Mask HURTS: raw correct, masked wrong
                    if raw_correct and not masked_correct:
                        mask_hurts.append({
                            "clip": stem, "fid": curr_fid,
                            "rc": rc, "mc": mc,
                            "err_raw": err_raw, "err_masked": err_masked
                        })
                        print(f"  [MASK HURTS!] {stem} f{curr_fid}: raw OK (err={err_raw:.1f} conf={rc:.3f}) -> masked KILLED (err={err_masked:.1f} conf={mc:.3f})", flush=True)

                else:
                    total_invisible += 1
                    # Case 2: False Alarm Blocked (no ball, raw >= 0.55, masked < 0.55)
                    if rc >= 0.55 and mc < 0.55:
                        case2_found.append({
                            "clip": stem, "fid": curr_fid,
                            "rc": rc, "mc": mc
                        })
                        print(f"  [CASE2 BLOCKED] {stem} f{curr_fid}: raw={rc:.3f} -> masked={mc:.3f} (FALSE ALARM BLOCKED!)", flush=True)

                frames.pop(0)
            fid += 1
        cap.release()

        if (ci + 1) % 20 == 0:
            print(f"  Progress: {ci+1}/{len(csv_list)} videos done...", flush=True)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total visible frames scanned: {total_visible}")
    print(f"Total invisible frames scanned: {total_invisible}")
    print(f"Raw correct detections (err<=5, conf>=0.55): {total_raw_correct}")
    print(f"Masked correct detections (err<=5, conf>=0.55): {total_masked_correct}")
    print(f"Difference (masked - raw): {total_masked_correct - total_raw_correct}")
    print()
    print(f"Case 1 (Peak Rescued - mask helps): {len(case1_found)}")
    print(f"Case 2 (False Alarm Blocked - mask helps): {len(case2_found)}")
    print(f"Mask HURTS (correct detection killed): {len(mask_hurts)}")
    print()

    if mask_hurts:
        print("--- Details of Mask HURTS cases ---")
        for h in mask_hurts[:20]:
            print(f"  {h['clip']} f{h['fid']}: raw conf={h['rc']:.3f} err={h['err_raw']:.1f} -> masked conf={h['mc']:.3f} err={h['err_masked']:.1f}")

    if case2_found:
        print("--- Details of Case 2 (False Alarm Blocked) ---")
        for c in case2_found[:20]:
            print(f"  {c['clip']} f{c['fid']}: raw={c['rc']:.3f} -> masked={c['mc']:.3f}")

    if case1_found:
        print("--- Details of Case 1 (Peak Rescued) ---")
        for c in case1_found[:20]:
            print(f"  {c['clip']} f{c['fid']}: raw_err={c['err_raw']:.1f} masked_err={c['err_masked']:.1f} rc={c['rc']:.3f} mc={c['mc']:.3f}")

if __name__ == "__main__":
    main()
