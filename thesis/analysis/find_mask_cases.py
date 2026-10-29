"""Search mask candidates offline, or observe the original realtime pipeline.

Place beside realtime_student_pc_with_events.py. Candidates require human review.
"""

# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from thesis.platforms.pc import student_realtime_events as rt


def write_jpeg(path, image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise OSError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--weights', default=r'checkpoints_v4_student\epoch_30.pth')
    p.add_argument('--mode', choices=['offline', 'realtime'], default='offline')
    p.add_argument('--event-dir', default='report_materials')
    p.add_argument('--mask', type=int, default=10)
    p.add_argument('--local-conf', type=float, default=0.55)
    p.add_argument('--global-conf', type=float, default=0.55)
    p.add_argument('--lost-duration', type=int, default=3)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--max-save', type=int, default=100)
    a = p.parse_args()
    if a.lost_duration < 1 or a.max_save < 0 or a.fps <= 0:
        p.error('lost-duration and fps must be positive; max-save must be nonnegative')
    root = Path(a.event_dir) / datetime.now().strftime('mask_search_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise FileNotFoundError(a.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    config = vars(a).copy()
    config.update(source_fps=source_fps, verified=False,
                  note='Same original mask and tracking thresholds. No ground truth verification. '
                  'Input frame ids are zero-based. Output target frame may be middle or latest; '
                  'inspect all three input images. Spatially distinct retained peak is not '
                  'necessarily the second-highest local maximum.')
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    fields = ['frame_id', 'source_time_s', 'scale', 'category', 'raw_conf',
              'masked_conf', 'raw_x', 'raw_y', 'masked_x', 'masked_y',
              'shift_px', 'mask_at_raw_peak', 'threshold', 'artifact']
    counts = {'case1_candidate': 0, 'case2_candidate': 0, 'other': 0}
    saved = {'case1_candidate': 0, 'case2_candidate': 0}
    stream = (root / 'all_frames.csv').open('w', newline='', encoding='utf-8-sig')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()

    def observe(logger, f_id, scale_tag, bgr_crop_or_img, resized_mask,
                raw_heatmap, masked_heatmap, conf_thresh, counts_dict,
                max_per_cat):
        rx, ry, rc = rt.get_shuttle_center_local(raw_heatmap)
        mx, my, mc = rt.get_shuttle_center_local(masked_heatmap)
        shift = float(np.hypot(rx - mx, ry - my))
        removed = float(resized_mask[ry, rx]) == 0.0
        min_shift = 6 if scale_tag.startswith('LOCAL') else 12
        category = 'other'
        if removed and shift >= min_shift and mc >= conf_thresh:
            category = 'case1_candidate'
        elif rc >= conf_thresh and mc < conf_thresh:
            category = 'case2_candidate'
        counts[category] += 1
        artifact = ''
        if category != 'other' and saved[category] < a.max_save:
            dest = root / category / f'{f_id:06d}_{scale_tag}'
            dest.mkdir(parents=True)
            # Neutral labels: the program cannot certify which peak is the ball.
            panels = []
            for title, data in [('Input (latest frame)', bgr_crop_or_img),
                                ('Motion mask', resized_mask),
                                (f'Raw peak {rc:.3f}', raw_heatmap),
                                (f'Masked peak {mc:.3f}', masked_heatmap)]:
                if data.ndim == 2:
                    tile = cv2.applyColorMap(np.clip(data * 255, 0, 255).astype('uint8'), cv2.COLORMAP_JET)
                else:
                    tile = data.copy()
                tile = cv2.resize(tile, (512, 288))
                canvas = np.zeros((324, 512, 3), np.uint8)
                canvas[36:] = tile
                cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)
                panels.append(canvas)
            write_jpeg(dest / 'comparison.jpg',
                        np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
            np.savez_compressed(dest / 'heatmaps.npz', raw=raw_heatmap,
                                masked=masked_heatmap, mask=resized_mask)
            if a.mode == 'offline':
                for index, frame in enumerate(frames):
                    write_jpeg(dest / f'input_{index}_frame_{f_id-2+index:06d}.jpg', frame)
            artifact = str(dest.relative_to(root))
            saved[category] += 1
            print(f'[CANDIDATE] {category} frame={f_id} {scale_tag} {rc:.3f}->{mc:.3f}')
        writer.writerow(dict(zip(fields, [f_id, f_id / source_fps if source_fps > 0 else '',
            scale_tag, category, rc, mc, rx, ry, mx, my, shift,
            float(resized_mask[ry, rx]), conf_thresh, artifact])))
        stream.flush()

    try:
        if a.mode == 'realtime':
            # Run the original three-thread pipeline, including its frame dropping.
            rt.evaluate_and_save_mask_sample = observe
            sys.argv = [sys.argv[0], '--video', a.video, '--weights', a.weights,
                        '--event-dir', str(root / 'realtime'), '--mask', str(a.mask),
                        '--local-conf', str(a.local_conf), '--global-conf', str(a.global_conf),
                        '--lost-duration', str(a.lost_duration), '--fps', str(a.fps)]
            rt.main()
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = rt.MiniROIUNet().to(device)
            model.load_state_dict(torch.load(a.weights, map_location=device))
            model.eval()
            cap = cv2.VideoCapture(a.video)
            frames = deque(maxlen=3)
            lost, misses, kf = True, 0, None
            fid = -1
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    fid += 1
                    frame = cv2.resize(frame, (rt.W_ORIG, rt.H_ORIG))
                    frames.append(cv2.resize(frame, (rt.W_TRAIN, rt.H_TRAIN)))
                    if len(frames) < 3:
                        continue

                    def infer(images, scale, threshold):
                        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images[:2]]
                        _, mask = cv2.threshold(cv2.absdiff(gray[1], gray[0]), a.mask, 255, cv2.THRESH_BINARY)
                        x = np.concatenate(images, axis=-1).astype(np.float32) / 255
                        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            raw = model(x).squeeze().cpu().numpy()
                        mask = cv2.resize(mask.astype(np.float32) / 255, (raw.shape[1], raw.shape[0]))
                        masked = raw * mask
                        observe(None, fid, scale, images[2], mask, raw, masked, threshold, {}, 0)
                        return rt.get_shuttle_center_local(masked)

                    if not lost and kf is not None:
                        pred = kf.predict()
                        px, py = int(pred[0, 0]), int(pred[1, 0])
                        crops = [rt.crop_and_pad(im, px, py, rt.ROI_SIZE)[0] for im in frames]
                        x, y, conf = infer(crops, 'LOCAL_128', a.local_conf)
                        if conf >= a.local_conf:
                            gx, gy = px - rt.ROI_SIZE // 2 + x, py - rt.ROI_SIZE // 2 + y
                            kf.correct(np.array([[gx], [gy]], dtype=np.float32))
                            misses = 0
                        else:
                            misses += 1
                            lost = misses >= a.lost_duration
                    if lost:
                        x, y, conf = infer(list(frames), 'GLOBAL_512', a.global_conf)
                        if conf > a.global_conf:
                            kf = rt.init_kalman_filter(x, y)
                            lost, misses = False, 0
                    if fid % 300 == 0:
                        print(f'[PROGRESS] frame={fid} counts={counts}')
            finally:
                cap.release()
    finally:
        stream.close()
        (root / 'summary.json').write_text(json.dumps(dict(counts=counts, saved=saved), indent=2), encoding='utf-8')
        print(f'Results: {root.resolve()}')


if __name__ == '__main__':
    main()
