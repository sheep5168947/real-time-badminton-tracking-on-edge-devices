# -*- coding: utf-8 -*-
"""
Synced PC version of Dual-Resolution MiniROI Tracker (aligned with Nano logic).
Features:
1. Synced algorithm with Nano (no top-60px suppression, fixed double-sigmoid bug, ROI/Lost box rendering).
2. Updated default parameters: mask=10, local_conf=0.55, global_conf=0.55, lost_duration=3.
3. Automatic logging of consecutive frames for KF Rescue Success vs. KF Rescue Failed.
4. Categorized Spatial-Temporal Motion Mask comparison generator:
   - 1_peak_rescued (Wrong background peak -> Corrected to true shuttlecock peak)
   - 2_false_alarm_blocked (Background false positive >= thresh -> Suppressed below thresh)
   - 3_noise_cleaned_same_peak (True peak preserved e.g. 0.88 -> 0.88, while background noise is killed)
5. Every run creates a unique timestamped folder so previous runs are NEVER overwritten.
"""

# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[3]))


import argparse
import csv
import json
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError as exc:
    raise ImportError(
        "Cannot import MiniROIUNet. Make sure student_model.py is beside this script."
    ) from exc


# ---------------------------------------------------------------------------
# Model and video dimensions
# ---------------------------------------------------------------------------
W_ORIG, H_ORIG = 1280, 720
W_TRAIN, H_TRAIN = 512, 288
ROI_SIZE = 128

# Shared memory between producer, inference, and display threads (Synced with Nano)
shared_frame_id = -1
shared_frame_data = None
shared_pred_point = (0, 0)
shared_is_lost = True
shared_status = "Init"
frame_lock = threading.Lock()
bbox_lock = threading.Lock()
is_running = True


def parse_args():
    """Parse runtime parameters; defaults match the requested thesis configuration."""
    parser = argparse.ArgumentParser(
        description="Dual-resolution MiniROI tracker (PC-Nano Synced) with KF & Mask report generators"
    )
    parser.add_argument(
        "--video",
        default=r"dataset\raw\part1\match9\video\1_04_05.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--weights",
        default=r"checkpoints_v4_student\epoch_30.pth",
        help="MiniROIUNet checkpoint path",
    )
    parser.add_argument(
        "--output-video",
        default="student_realtime_pc_synced.mp4",
        help="Annotated output video filename (saved inside the timestamped run folder)",
    )
    parser.add_argument(
        "--event-dir",
        default="report_materials",
        help="Root directory for KF rescue events and Spatial-Temporal Mask comparisons",
    )
    parser.add_argument(
        "--mask", type=int, default=10, help="Frame-difference motion mask threshold (default: 10)"
    )
    parser.add_argument(
        "--local-conf", type=float, default=0.55, help="Local heatmap confidence threshold (default: 0.55)"
    )
    parser.add_argument(
        "--global-conf", type=float, default=0.55, help="Global heatmap confidence threshold (default: 0.55)"
    )
    parser.add_argument(
        "--lost-duration",
        type=int,
        default=3,
        help="Consecutive local misses tolerated before global search (default: 3)",
    )
    parser.add_argument(
        "--max-mask-per-category",
        type=int,
        default=25,
        help="Maximum number of Mask comparison figures to save per category",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Playback/output pacing FPS"
    )
    return parser.parse_args()


def get_shuttle_center_local(heatmap):
    """Return the peak location and confidence from a 2D heatmap (Synced with Nano)."""
    max_val = float(np.max(heatmap))
    if max_val <= 0.0:
        return 0, 0, 0.0
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return int(idx[-1]), int(idx[-2]), max_val


def crop_and_pad(img, center_x, center_y, size):
    """Crop a fixed-size region around a center, padding out-of-frame areas."""
    h, w = img.shape[:2]
    half = size // 2
    x_min, x_max = int(center_x - half), int(center_x + half)
    y_min, y_max = int(center_y - half), int(center_y + half)

    valid_x_min = max(0, x_min)
    valid_x_max = min(w, x_max)
    valid_y_min = max(0, y_min)
    valid_y_max = min(h, y_max)

    valid_x_max = max(valid_x_min, valid_x_max)
    valid_y_max = max(valid_y_min, valid_y_max)
    cropped = img[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

    if cropped.shape[0] != size or cropped.shape[1] != size:
        if img.ndim == 3:
            padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((size, size), dtype=img.dtype)
        offset_x = valid_x_min - x_min
        offset_y = valid_y_min - y_min
        target_x1 = max(0, offset_x)
        target_y1 = max(0, offset_y)
        source_x1 = max(0, -offset_x)
        source_y1 = max(0, -offset_y)
        copy_w = min(cropped.shape[1] - source_x1, size - target_x1)
        copy_h = min(cropped.shape[0] - source_y1, size - target_y1)
        if copy_w > 0 and copy_h > 0:
            padded[
                target_y1 : target_y1 + copy_h,
                target_x1 : target_x1 + copy_w,
            ] = cropped[
                source_y1 : source_y1 + copy_h,
                source_x1 : source_x1 + copy_w,
            ]
        return padded, x_min, y_min

    return cropped, x_min, y_min


def init_kalman_filter(init_x, init_y):
    """Initialize a constant-velocity Kalman filter for shuttlecock tracking."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.statePost = np.array(
        [[float(init_x)], [float(init_y)], [0.0], [0.0]], dtype=np.float32
    )
    return kf


# ---------------------------------------------------------------------------
# Visualization & Report Material Helpers
# ---------------------------------------------------------------------------
def save_spatial_temporal_mask_comparison(
    save_dir,
    frame_id,
    scale_tag,
    case_type,
    bgr_img,
    motion_mask,
    raw_heatmap,
    masked_heatmap,
    noise_x=0,
    noise_y=0,
    max_noise_killed=0.0,
):
    """
    Generate a 2x2 visual comparison of the Spatial-Temporal Mask effect:
    [Top-Left] Input RGB (Frame or ROI)     [Top-Right] Binary Motion Mask
    [Bottom-Left] Raw Heatmap (Before Mask) [Bottom-Right] Masked Heatmap (After Mask)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if scale_tag.startswith("LOCAL"):
        tile_w, tile_h = 320, 320
    else:
        tile_w, tile_h = 512, 288

    # 1. Input RGB image
    rgb_vis = cv2.resize(bgr_img, (tile_w, tile_h))

    # 2. Motion Mask (0~1 float -> 0~255 grayscale -> BGR)
    mask_u8 = np.clip(motion_mask * 255.0, 0, 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(
        cv2.resize(mask_u8, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    # 3. Raw Heatmap (Before Spatial-Temporal Mask)
    raw_u8 = np.clip(raw_heatmap * 255.0, 0, 255).astype(np.uint8)
    raw_color = cv2.applyColorMap(cv2.resize(raw_u8, (tile_w, tile_h)), cv2.COLORMAP_JET)
    raw_blend = cv2.addWeighted(rgb_vis, 0.35, raw_color, 0.65, 0)

    # 4. Masked Heatmap (After Spatial-Temporal Mask)
    masked_u8 = np.clip(masked_heatmap * 255.0, 0, 255).astype(np.uint8)
    masked_color = cv2.applyColorMap(cv2.resize(masked_u8, (tile_w, tile_h)), cv2.COLORMAP_JET)
    masked_blend = cv2.addWeighted(rgb_vis, 0.35, masked_color, 0.65, 0)

    # Peak coordinates and values
    rx, ry, r_conf = get_shuttle_center_local(raw_heatmap)
    mx, my, m_conf = get_shuttle_center_local(masked_heatmap)
    h_hm, w_hm = raw_heatmap.shape[:2]

    # Annotate Panel 3 (Before Mask) & Panel 4 (After Mask) according to case_type
    if case_type == "PEAK_RESCUED":
        # Raw peak was a false positive in the background!
        if r_conf > 0:
            prx, pry = int(rx * tile_w / w_hm), int(ry * tile_h / h_hm)
            cv2.drawMarker(raw_blend, (prx, pry), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(
                raw_blend,
                f"Fake Peak ({r_conf:.2f})",
                (max(4, prx - 45), max(18, pry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        if m_conf > 0:
            pmx, pmy = int(mx * tile_w / w_hm), int(my * tile_h / h_hm)
            # Show where the true shuttlecock was hiding in Panel 3
            cv2.circle(raw_blend, (pmx, pmy), 10, (0, 255, 0), 2)
            # Show the rescued peak in Panel 4
            cv2.drawMarker(masked_blend, (pmx, pmy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                masked_blend,
                f"Rescued True ({m_conf:.2f})",
                (max(4, pmx - 50), max(18, pmy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        title_3 = f"3. Before Mask (Fake Max:{r_conf:.2f} -> Wrong!)"
        title_4 = f"4. After Mask (True Max:{m_conf:.2f} -> Rescued!)"

    elif case_type == "FALSE_ALARM_BLOCKED":
        if r_conf > 0:
            prx, pry = int(rx * tile_w / w_hm), int(ry * tile_h / h_hm)
            cv2.drawMarker(raw_blend, (prx, pry), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(
                raw_blend,
                f"False Alarm ({r_conf:.2f})",
                (max(4, prx - 50), max(18, pry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        title_3 = f"3. Before Mask (False Alarm Max:{r_conf:.2f})"
        title_4 = f"4. After Mask (Suppressed Max:{m_conf:.2f})"

    else:
        # NOISE_CLEANED_SAME_PEAK (e.g., 0.88 -> 0.88 while killing background noise)
        if max_noise_killed > 0.15:
            pnx, pny = int(noise_x * tile_w / w_hm), int(noise_y * tile_h / h_hm)
            cv2.drawMarker(raw_blend, (pnx, pny), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(
                raw_blend,
                f"BgNoise:{max_noise_killed:.2f}",
                (max(4, pnx - 40), max(18, pny - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        if r_conf > 0:
            prx, pry = int(rx * tile_w / w_hm), int(ry * tile_h / h_hm)
            cv2.drawMarker(raw_blend, (prx, pry), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        if m_conf > 0:
            pmx, pmy = int(mx * tile_w / w_hm), int(my * tile_h / h_hm)
            cv2.drawMarker(masked_blend, (pmx, pmy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        title_3 = f"3. Before Mask (Max:{r_conf:.2f} | BgNoise:{max_noise_killed:.2f})"
        title_4 = f"4. After Mask (Max:{m_conf:.2f} | BgNoise->0.00)"

    def add_header(img, title):
        header_h = 34
        canvas = np.zeros((img.shape[0] + header_h, img.shape[1], 3), dtype=np.uint8)
        canvas[:header_h, :] = (30, 30, 30)
        canvas[header_h:, :] = img
        cv2.putText(
            canvas,
            title,
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    t1 = add_header(rgb_vis, f"1. Input ({scale_tag}) | Frame {frame_id}")
    t2 = add_header(mask_vis, f"2. Spatial-Temporal Mask (diff >= thresh)")
    t3 = add_header(raw_blend, title_3)
    t4 = add_header(masked_blend, title_4)

    top_row = np.hstack([t1, t2])
    bot_row = np.hstack([t3, t4])
    grid = np.vstack([top_row, bot_row])

    out_file = (
        save_dir
        / f"frame_{frame_id:06d}_{scale_tag}_{case_type}_raw{r_conf:.2f}_masked{m_conf:.2f}_noise{max_noise_killed:.2f}.jpg"
    )
    cv2.imwrite(str(out_file), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])


def draw_event_overlay(
    frame,
    frame_id,
    label,
    mode,
    roi_center=None,
    detected_center=None,
    confidence=None,
    miss_count=None,
):
    """Annotate a source-resolution frame for visual inspection of a KF rescue event."""
    canvas = frame.copy()
    scale_x = W_ORIG / float(W_TRAIN)
    scale_y = H_ORIG / float(H_TRAIN)

    if roi_center is not None:
        center_x, center_y = roi_center
        half = ROI_SIZE // 2
        x1 = int((center_x - half) * scale_x)
        y1 = int((center_y - half) * scale_y)
        x2 = int((center_x + half) * scale_x)
        y2 = int((center_y + half) * scale_y)
        color = (0, 255, 0) if "SUCCESS" in label or "TRACKING" in label else (0, 220, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.drawMarker(
            canvas,
            (int(center_x * scale_x), int(center_y * scale_y)),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )

    if detected_center is not None:
        detected_x, detected_y = detected_center
        cv2.circle(
            canvas,
            (int(detected_x * scale_x), int(detected_y * scale_y)),
            10,
            (0, 0, 255),
            -1,
        )

    details = [f"Frame: {frame_id}", f"State: {mode}", f"Event: {label}"]
    if confidence is not None:
        details.append(f"Confidence: {confidence:.3f}")
    if miss_count is not None:
        details.append(f"Consecutive Local Misses: {miss_count}")

    line_height = 28
    strip_height = line_height * len(details) + 16
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (520, strip_height), (15, 25, 40), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    for index, line in enumerate(details):
        cv2.putText(
            canvas,
            line,
            (16, 28 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def create_contact_sheet(image_paths, output_path, max_images=12):
    """Create a slide-friendly contact sheet from an event's consecutive frames."""
    readable = []
    for image_path in image_paths[:max_images]:
        image = cv2.imread(str(image_path))
        if image is not None:
            readable.append((image_path, cv2.resize(image, (480, 270))))

    if not readable:
        return

    columns = 3
    rows = (len(readable) + columns - 1) // columns
    margin = 16
    label_height = 28
    cell_w, cell_h = 480, 270 + label_height
    sheet = np.full(
        (
            rows * cell_h + (rows + 1) * margin,
            columns * cell_w + (columns + 1) * margin,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for index, (image_path, image) in enumerate(readable):
        row, col = divmod(index, columns)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        sheet[y : y + 270, x : x + cell_w] = image
        title = Path(image_path).stem
        cv2.putText(
            sheet,
            title[:52],
            (x + 4, y + 270 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 30, 50),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), sheet)


class RescueEventLogger:
    """
    Log consecutive frames for KF Rescue episodes and organize Mask Comparisons into:
    - mask_comparisons/1_peak_rescued/
    - mask_comparisons/2_false_alarm_blocked/
    - mask_comparisons/3_noise_cleaned_same_peak/
    """

    CSV_FIELDS = [
        "event_id",
        "category",
        "start_frame_id",
        "end_frame_id",
        "local_misses_count",
        "outcome",
        "last_local_confidence",
        "recovery_confidence",
        "event_folder",
        "contact_sheet",
    ]

    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        self.success_dir = self.run_dir / "kf_rescue_success"
        self.failed_dir = self.run_dir / "kf_rescue_failed"
        self.mask_dir = self.run_dir / "mask_comparisons"
        self.mask_rescued_dir = self.mask_dir / "1_peak_rescued"
        self.mask_blocked_dir = self.mask_dir / "2_false_alarm_blocked"
        self.mask_cleaned_dir = self.mask_dir / "3_noise_cleaned_same_peak"
        self.temp_dir = self.run_dir / "_active_tmp"

        for d in (
            self.success_dir,
            self.failed_dir,
            self.mask_rescued_dir,
            self.mask_blocked_dir,
            self.mask_cleaned_dir,
            self.temp_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.event_counter = 0
        self.csv_path = self.run_dir / "kf_rescue_summary.csv"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.CSV_FIELDS)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        self.active_event = None
        self._write_config()
        print(f"[INFO] Report materials directory (New Run): {self.run_dir.resolve()}")

    def _write_config(self):
        config_path = self.run_dir / "run_config.json"
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config, file, ensure_ascii=False, indent=2)

    def _save_frame(
        self,
        event,
        frame,
        frame_id,
        tag,
        mode,
        roi_center=None,
        detected_center=None,
        confidence=None,
        miss_count=None,
    ):
        annotated = draw_event_overlay(
            frame=frame,
            frame_id=frame_id,
            label=tag,
            mode=mode,
            roi_center=roi_center,
            detected_center=detected_center,
            confidence=confidence,
            miss_count=miss_count,
        )
        filename = f"{len(event['frame_paths']) + 1:02d}_frame_{frame_id:06d}_{tag}.jpg"
        output_path = event["folder"] / filename
        cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        event["frame_paths"].append(output_path)
        return output_path

    def begin(self, frame_id, history):
        """Start an episode on the first local miss, saving 2 preceding context frames."""
        self.event_counter += 1
        event_id = f"event_{self.event_counter:04d}"
        event_folder = self.temp_dir / event_id
        event_folder.mkdir(parents=True, exist_ok=True)
        event = {
            "event_id": event_id,
            "folder": event_folder,
            "frame_paths": [],
            "start_frame_id": frame_id,
            "local_misses": 0,
            "last_local_confidence": None,
            "recovery_confidence": None,
            "post_fail_frames_saved": 0,
        }

        for old_f_id, old_frame, old_roi, old_det, old_conf in list(history)[-2:]:
            self._save_frame(
                event,
                old_frame,
                old_f_id,
                "PRE_MISS_TRACKING",
                mode="LOCAL_TRACK_128",
                roi_center=old_roi,
                detected_center=old_det,
                confidence=old_conf,
                miss_count=0,
            )

        self.active_event = event
        return event

    def record_local_miss(self, frame, frame_id, roi_center, confidence, miss_count, is_lost_now):
        event = self.active_event
        if event is None:
            return
        event["local_misses"] = miss_count
        event["last_local_confidence"] = float(confidence)
        tag = f"KF_MISS_{miss_count}_TRIGGER_LOST" if is_lost_now else f"KF_MISS_{miss_count}"
        self._save_frame(
            event,
            frame,
            frame_id,
            tag,
            mode="KF_PREDICT_128",
            roi_center=roi_center,
            confidence=confidence,
            miss_count=miss_count,
        )

    def record_local_rescue_success(self, frame, frame_id, roi_center, detected_center, confidence):
        event = self.active_event
        if event is None:
            return
        event["recovery_confidence"] = float(confidence)
        self._save_frame(
            event,
            frame,
            frame_id,
            "KF_RESCUE_SUCCESS",
            mode="LOCAL_TRACK_RECOVERED",
            roi_center=roi_center,
            detected_center=detected_center,
            confidence=confidence,
            miss_count=event["local_misses"],
        )
        self.finish(
            category="SUCCESS",
            outcome="KF_LOCAL_RESCUE_SUCCESS",
            end_frame_id=frame_id,
        )

    def record_global_after_kf_failure(self, frame, frame_id, confidence, detected_center=None, recovered=False):
        event = self.active_event
        if event is None:
            return
        event["post_fail_frames_saved"] += 1
        event["recovery_confidence"] = float(confidence)

        if recovered:
            self._save_frame(
                event,
                frame,
                frame_id,
                "KF_FAILED_THEN_GLOBAL_RECOVERED",
                mode="GLOBAL_RECOVERY_512",
                detected_center=detected_center,
                confidence=confidence,
                miss_count=event["local_misses"],
            )
            self.finish(
                category="FAILED",
                outcome="KF_RESCUE_FAILED_GLOBAL_RECOVERED",
                end_frame_id=frame_id,
            )
        else:
            self._save_frame(
                event,
                frame,
                frame_id,
                f"KF_FAILED_GLOBAL_SEARCH_{event['post_fail_frames_saved']}",
                mode="GLOBAL_SEARCH_512",
                confidence=confidence,
                miss_count=event["local_misses"],
            )
            if event["post_fail_frames_saved"] >= 2:
                self.finish(
                    category="FAILED",
                    outcome="KF_RESCUE_FAILED_STILL_LOST",
                    end_frame_id=frame_id,
                )

    def finish(self, category, outcome, end_frame_id):
        event = self.active_event
        if event is None:
            return

        target_parent = self.success_dir if category == "SUCCESS" else self.failed_dir
        final_folder_name = (
            f"{event['event_id']}_frames_{event['start_frame_id']}_to_{end_frame_id}"
            f"_misses_{event['local_misses']}"
        )
        final_folder = target_parent / final_folder_name
        shutil.move(str(event["folder"]), str(final_folder))

        moved_frames = sorted(final_folder.glob("*.jpg"))
        contact_sheet_path = final_folder / "contact_sheet.jpg"
        create_contact_sheet(moved_frames, contact_sheet_path)

        metadata = {
            "event_id": event["event_id"],
            "category": category,
            "outcome": outcome,
            "start_frame_id": event["start_frame_id"],
            "end_frame_id": end_frame_id,
            "local_misses_count": event["local_misses"],
            "last_local_confidence": event["last_local_confidence"],
            "recovery_confidence": event["recovery_confidence"],
            "settings": self.config,
        }
        with (final_folder / "event_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.csv_writer.writerow(
            {
                "event_id": event["event_id"],
                "category": category,
                "start_frame_id": event["start_frame_id"],
                "end_frame_id": end_frame_id,
                "local_misses_count": event["local_misses"],
                "outcome": outcome,
                "last_local_confidence": event["last_local_confidence"],
                "recovery_confidence": event["recovery_confidence"],
                "event_folder": str(final_folder.resolve()),
                "contact_sheet": str(contact_sheet_path.resolve()),
            }
        )
        self.csv_file.flush()
        print(
            f"[KF-EVENT] {event['event_id']} | Category: {category} ({outcome}) | "
            f"Misses: {event['local_misses']} | Saved to: {final_folder.name}"
        )
        self.active_event = None

    def close(self, final_frame_id):
        if self.active_event is not None:
            cat = "FAILED" if self.active_event["local_misses"] >= self.config["lost_duration"] else "UNRESOLVED"
            self.finish(cat, "VIDEO_ENDED_DURING_EVENT", final_frame_id)
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.csv_file.close()


# ---------------------------------------------------------------------------
# Thread 1: Video Producer
# ---------------------------------------------------------------------------
def camera_thread(video_path, fps):
    global shared_frame_id, shared_frame_data, is_running
    print("[INFO] [Thread 1] Video producer started.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input video: {video_path}")
        is_running = False
        return

    frame_delay = 1.0 / fps if fps > 0 else 0.0
    local_frame_id = 0

    while is_running:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            is_running = False
            break

        frame_resized = cv2.resize(frame, (W_ORIG, H_ORIG))
        with frame_lock:
            shared_frame_data = frame_resized.copy()
            shared_frame_id = local_frame_id
        local_frame_id += 1

        read_time = time.perf_counter() - start_time
        sleep_time = frame_delay - read_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    print("[INFO] [Thread 1] Video producer terminated.")


# ---------------------------------------------------------------------------
# Helper: Evaluate & Save Categorized Mask Comparisons
# ---------------------------------------------------------------------------
def evaluate_and_save_mask_sample(
    logger,
    f_id,
    scale_tag,
    bgr_crop_or_img,
    resized_mask,
    raw_heatmap,
    masked_heatmap,
    conf_thresh,
    counts_dict,
    max_per_cat,
):
    """
    Classify the current frame's Spatial-Temporal Mask effect into one of 3 categories:
    1. PEAK_RESCUED: Raw peak was on static background noise, Masked peak shifted to true shuttlecock!
    2. FALSE_ALARM_BLOCKED: Raw peak >= conf_thresh (false alarm), Masked peak < conf_thresh (blocked)!
    3. NOISE_CLEANED_SAME_PEAK: Peak stayed on true shuttlecock (e.g. 0.88 -> 0.88), while strong background noise was zeroed out.
    """
    raw_x, raw_y, raw_conf = get_shuttle_center_local(raw_heatmap)
    masked_x, masked_y, masked_conf = get_shuttle_center_local(masked_heatmap)

    # Heatmap of static regions that were zeroed out by the motion mask
    suppressed_map = raw_heatmap * (1.0 - resized_mask)
    noise_x, noise_y, max_noise_killed = get_shuttle_center_local(suppressed_map)

    dist_shift = float(np.hypot(raw_x - masked_x, raw_y - masked_y))
    min_shift_px = 6.0 if scale_tag.startswith("LOCAL") else 12.0

    # Category 1: Golden Case - Peak Rescued (False peak -> True shuttlecock peak)
    if dist_shift >= min_shift_px and masked_conf >= conf_thresh:
        if counts_dict["rescued"] < max_per_cat:
            save_spatial_temporal_mask_comparison(
                logger.mask_rescued_dir,
                f_id,
                scale_tag,
                "PEAK_RESCUED",
                bgr_crop_or_img,
                resized_mask,
                raw_heatmap,
                masked_heatmap,
                noise_x=raw_x,
                noise_y=raw_y,
                max_noise_killed=raw_conf,
            )
            counts_dict["rescued"] += 1
            print(
                f"[MASK-GOLDEN] Frame {f_id} ({scale_tag}) | 1_PEAK_RESCUED: "
                f"Fake Peak ({raw_conf:.2f}) -> True Peak ({masked_conf:.2f})"
            )
        return

    # Category 2: Golden Case - False Alarm Blocked (Static background > thresh -> suppressed < thresh)
    if raw_conf >= conf_thresh and masked_conf < conf_thresh:
        if counts_dict["blocked"] < max_per_cat:
            save_spatial_temporal_mask_comparison(
                logger.mask_blocked_dir,
                f_id,
                scale_tag,
                "FALSE_ALARM_BLOCKED",
                bgr_crop_or_img,
                resized_mask,
                raw_heatmap,
                masked_heatmap,
                noise_x=raw_x,
                noise_y=raw_y,
                max_noise_killed=raw_conf,
            )
            counts_dict["blocked"] += 1
            print(
                f"[MASK-GOLDEN] Frame {f_id} ({scale_tag}) | 2_FALSE_ALARM_BLOCKED: "
                f"Raw ({raw_conf:.2f}) >= {conf_thresh} -> Masked ({masked_conf:.2f})"
            )
        return

    # Category 3: Same Peak Preserved (e.g. 0.88 -> 0.88) + Background Noise Cleaned
    noise_thresh = 0.25 if scale_tag.startswith("LOCAL") else 0.35
    if masked_conf >= conf_thresh and max_noise_killed >= noise_thresh:
        if counts_dict["cleaned"] < max_per_cat:
            save_spatial_temporal_mask_comparison(
                logger.mask_cleaned_dir,
                f_id,
                scale_tag,
                "NOISE_CLEANED",
                bgr_crop_or_img,
                resized_mask,
                raw_heatmap,
                masked_heatmap,
                noise_x=noise_x,
                noise_y=noise_y,
                max_noise_killed=max_noise_killed,
            )
            counts_dict["cleaned"] += 1


# ---------------------------------------------------------------------------
# Thread 2: AI Inference (Synced with Nano + Categorized Report Logging)
# ---------------------------------------------------------------------------
def ai_thread(
    weights_path,
    run_dir,
    mask_threshold,
    local_conf,
    global_conf,
    lost_duration,
    max_mask_per_category,
):
    global shared_pred_point, shared_is_lost, shared_status, is_running
    print("[INFO] [Thread 2] AI inference started (PyTorch backend, Nano-synced logic).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inference device: {device}")
    model = MiniROIUNet().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    logger = RescueEventLogger(
        run_dir,
        config={
            "mask_threshold": mask_threshold,
            "local_confidence_threshold": local_conf,
            "global_confidence_threshold": global_conf,
            "lost_duration": lost_duration,
            "roi_size": ROI_SIZE,
            "top_60px_suppression": False,
        },
    )

    ai_frame_queue = deque(maxlen=3)
    tracking_history = deque(maxlen=4)
    last_processed_id = -1

    # State Machine Variables (Synced with realtime_student_nano.py)
    is_lost = True
    miss_count = 0
    kf = None
    last_x_512, last_y_512 = W_TRAIN // 2, H_TRAIN // 2

    stats_total_frames = 0
    stats_track_128 = 0
    stats_global_rec = 0
    mask_counts = {"rescued": 0, "blocked": 0, "cleaned": 0}

    while is_running:
        with frame_lock:
            f_id = shared_frame_id
            original_frame = None if shared_frame_data is None else shared_frame_data.copy()

        if f_id <= last_processed_id or original_frame is None:
            time.sleep(0.001)
            continue

        last_processed_id = f_id
        start_time = time.perf_counter()

        f_train = cv2.resize(original_frame, (W_TRAIN, H_TRAIN))
        ai_frame_queue.append(f_train)

        if len(ai_frame_queue) < 3:
            continue

        img0, img1, img2 = ai_frame_queue[0], ai_frame_queue[1], ai_frame_queue[2]
        student_found = False
        current_status = "Processing..."
        stats_total_frames += 1

        # ==========================================
        # Phase 1: Local ROI Tracking (128x128)
        # ==========================================
        if not is_lost and kf is not None:
            predicted_state = kf.predict()
            pred_x, pred_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])
            roi_center = (pred_x, pred_y)

            c0, _, _ = crop_and_pad(img0, pred_x, pred_y, ROI_SIZE)
            c1, _, _ = crop_and_pad(img1, pred_x, pred_y, ROI_SIZE)
            c2, _, _ = crop_and_pad(img2, pred_x, pred_y, ROI_SIZE)

            g0 = cv2.cvtColor(c0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(
                cv2.absdiff(g1, g0), mask_threshold, 255, cv2.THRESH_BINARY
            )
            motion_mask = motion_mask.astype(np.float32) / 255.0

            imgs = np.concatenate((c0, c1, c2), axis=-1).astype(np.float32) / 255.0
            tensor_x = torch.from_numpy(np.transpose(imgs, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap = model(tensor_x).squeeze().cpu().numpy()

            resized_mask = cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            heatmap_masked = heatmap * resized_mask
            lx, ly, conf = get_shuttle_center_local(heatmap_masked)

            # Evaluate and save categorized Spatial-Temporal Mask comparisons (Local 128)
            evaluate_and_save_mask_sample(
                logger=logger,
                f_id=f_id,
                scale_tag="LOCAL_128",
                bgr_crop_or_img=c2,
                resized_mask=resized_mask,
                raw_heatmap=heatmap,
                masked_heatmap=heatmap_masked,
                conf_thresh=local_conf,
                counts_dict=mask_counts,
                max_per_cat=max_mask_per_category,
            )

            if conf >= local_conf:
                gx = pred_x - ROI_SIZE // 2 + lx
                gy = pred_y - ROI_SIZE // 2 + ly
                kf.correct(np.array([[float(gx)], [float(gy)]], dtype=np.float32))
                last_x_512, last_y_512 = gx, gy
                student_found = True
                miss_count = 0
                current_status = f"Track 128 (Conf:{conf:.2f})"
                stats_track_128 += 1

                if logger.active_event is not None:
                    logger.record_local_rescue_success(
                        original_frame,
                        f_id,
                        roi_center=roi_center,
                        detected_center=(gx, gy),
                        confidence=conf,
                    )

                tracking_history.append((f_id, original_frame.copy(), roi_center, (gx, gy), conf))
            else:
                miss_count += 1
                if logger.active_event is None:
                    logger.begin(f_id, tracking_history)

                if miss_count >= lost_duration:
                    is_lost = True
                    current_status = f"KF Lost ({miss_count}/{lost_duration})"
                else:
                    last_x_512, last_y_512 = pred_x, pred_y
                    current_status = f"KF Predict (Miss {miss_count}/{lost_duration}, Conf:{conf:.2f})"
                    stats_track_128 += 1

                logger.record_local_miss(
                    original_frame,
                    f_id,
                    roi_center=roi_center,
                    confidence=conf,
                    miss_count=miss_count,
                    is_lost_now=is_lost,
                )

        # ==========================================
        # Phase 2: Global Recovery (512x288)
        # ==========================================
        if is_lost:
            g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(
                cv2.absdiff(g1, g0), mask_threshold, 255, cv2.THRESH_BINARY
            )
            motion_mask = motion_mask.astype(np.float32) / 255.0

            imgs = np.concatenate((img0, img1, img2), axis=-1).astype(np.float32) / 255.0
            tensor_x = torch.from_numpy(np.transpose(imgs, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmap = model(tensor_x).squeeze().cpu().numpy()

            resized_mask = cv2.resize(motion_mask, (heatmap.shape[1], heatmap.shape[0]))
            heatmap_masked = heatmap * resized_mask
            gx, gy, conf = get_shuttle_center_local(heatmap_masked)

            # Evaluate and save categorized Spatial-Temporal Mask comparisons (Global 512)
            evaluate_and_save_mask_sample(
                logger=logger,
                f_id=f_id,
                scale_tag="GLOBAL_512",
                bgr_crop_or_img=img2,
                resized_mask=resized_mask,
                raw_heatmap=heatmap,
                masked_heatmap=heatmap_masked,
                conf_thresh=global_conf,
                counts_dict=mask_counts,
                max_per_cat=max_mask_per_category,
            )

            if conf > global_conf:
                kf = init_kalman_filter(gx, gy)
                last_x_512, last_y_512 = gx, gy
                is_lost = False
                miss_count = 0
                student_found = True
                current_status = f"Global Rec (Conf:{conf:.2f})"
                stats_global_rec += 1

                if logger.active_event is not None:
                    logger.record_global_after_kf_failure(
                        original_frame,
                        f_id,
                        confidence=conf,
                        detected_center=(gx, gy),
                        recovered=True,
                    )

                tracking_history.append((f_id, original_frame.copy(), (gx, gy), (gx, gy), conf))
            else:
                current_status = f"Global Search... (Conf:{conf:.2f})"
                stats_global_rec += 1
                if logger.active_event is not None:
                    logger.record_global_after_kf_failure(
                        original_frame,
                        f_id,
                        confidence=conf,
                        recovered=False,
                    )

        with bbox_lock:
            shared_pred_point = (
                (last_x_512, last_y_512)
                if (student_found or not is_lost)
                else (0, 0)
            )
            shared_is_lost = is_lost
            shared_status = current_status

        infer_time = (time.perf_counter() - start_time) * 1000.0
        if f_id % 15 == 0:
            print(
                f"[INFO] Frame {f_id} | Status: {current_status} | Latency: {infer_time:.2f} ms"
            )

    logger.close(last_processed_id)
    print("\n========================================")
    print("📊 128x128 模式 - 引擎觸發佔比與素材統計報告 (PC Synced)")
    print(f"總處理幀數: {stats_total_frames}")
    if stats_total_frames > 0:
        print(f"[128x128 局部追蹤] 佔比: {(stats_track_128 / stats_total_frames) * 100:.1f}%")
        print(f"[512x288 全域救援] 佔比: {(stats_global_rec / stats_total_frames) * 100:.1f}%")
    print("----------------------------------------")
    print(f"✨ Mask 黃金案例 1 [反敗為勝 Peak Rescued]:   {mask_counts['rescued']} 張 -> {logger.mask_rescued_dir}")
    print(f"🛡️ Mask 黃金案例 2 [攔截誤報 Alarm Blocked]: {mask_counts['blocked']} 張 -> {logger.mask_blocked_dir}")
    print(f"🧹 Mask 保真濾波 3 [真球不變 Noise Cleaned]: {mask_counts['cleaned']} 張 -> {logger.mask_cleaned_dir}")
    print(f"📁 KF 救援事件統計表: {logger.csv_path.resolve()}")
    print("========================================\n")


# ---------------------------------------------------------------------------
# Thread 3: Main Display & Video Writer (Synced with Nano UI)
# ---------------------------------------------------------------------------
def main():
    global is_running
    args = parse_args()

    print("[INFO] System Initialization (PC-Nano Synced Mode)...")
    print(
        f"[INFO] Parameters: mask={args.mask}, local_conf={args.local_conf}, "
        f"global_conf={args.global_conf}, lost_duration={args.lost_duration}"
    )

    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    # Create a unique timestamped run folder so previous runs and videos are NEVER overwritten
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.event_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = run_dir / Path(args.output_video).name
    out = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W_ORIG, H_ORIG)
    )

    t_camera = threading.Thread(target=camera_thread, args=(args.video, args.fps), daemon=True)
    t_ai = threading.Thread(
        target=ai_thread,
        args=(
            args.weights,
            run_dir,
            args.mask,
            args.local_conf,
            args.global_conf,
            args.lost_duration,
            args.max_mask_per_category,
        ),
        daemon=True,
    )

    t_camera.start()
    t_ai.start()

    frame_delay = 1.0 / args.fps if args.fps > 0 else 0.0
    last_drawn_id = -1
    scale_x = W_ORIG / float(W_TRAIN)
    scale_y = H_ORIG / float(H_TRAIN)

    while is_running or t_camera.is_alive() or t_ai.is_alive():
        start_time = time.perf_counter()

        with frame_lock:
            f_id = shared_frame_id
            frame_resized = None if shared_frame_data is None else shared_frame_data.copy()

        if frame_resized is None or f_id <= last_drawn_id:
            time.sleep(0.001)
            if not t_camera.is_alive() and not t_ai.is_alive():
                break
            continue

        last_drawn_id = f_id

        with bbox_lock:
            px, py = shared_pred_point
            is_lost_state = shared_is_lost
            status_text = shared_status

        if is_lost_state:
            cv2.rectangle(frame_resized, (0, 0), (W_ORIG - 1, H_ORIG - 1), (0, 0, 255), 5)
        else:
            box_w = int(ROI_SIZE * scale_x)
            box_h = int(ROI_SIZE * scale_y)
            top_left = (int(px * scale_x - box_w // 2), int(py * scale_y - box_h // 2))
            bottom_right = (int(px * scale_x + box_w // 2), int(py * scale_y + box_h // 2))

            is_predict_only = status_text.startswith("KF Predict")
            box_color = (0, 220, 255) if is_predict_only else (0, 255, 0)
            cv2.rectangle(frame_resized, top_left, bottom_right, box_color, 2)

            if px != 0 and py != 0:
                draw_x = int(px * scale_x)
                draw_y = int(py * scale_y)
                pt_color = (0, 220, 255) if is_predict_only else (0, 0, 255)
                cv2.circle(frame_resized, (draw_x, draw_y), 10, pt_color, -1)

        color = (0, 0, 255) if is_lost_state else (0, 255, 0)
        cv2.putText(
            frame_resized,
            "System: Dual-Res PyTorch + KF + MotionMask",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_resized,
            f"AI State: {status_text}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        out.write(frame_resized)

        process_time = time.perf_counter() - start_time
        sleep_time = frame_delay - process_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    is_running = False
    t_camera.join()
    t_ai.join()
    out.release()
    print(f"[INFO] System Shutdown Complete. Video saved to: {output_video_path.resolve()}")


if __name__ == "__main__":
    main()