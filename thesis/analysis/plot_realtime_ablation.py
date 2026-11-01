# -*- coding: utf-8 -*-
"""
Plotting Real-Time Hardware Ablation Results on Jetson Nano.
Visualizes:
1. Hardware Camera Frame Drop Rate (%)
2. Real-Time Tracking F1-Score & Recall (accounting for camera drops)
3. Effective Processing FPS
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

def plot_realtime(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    labels = []
    drop_rates = []
    f1_scores = []
    recalls = []
    effective_fps = []
    engine_fps = []

    for r in rows:
        name = r["Variant"]
        if "Pure Global" in name:
            short_name = "Native TrackNetV4\n(Pure Global 512)"
        elif "Proposed" in name:
            short_name = "Proposed MiniROI\n(KF + Mask)"
        elif "No KF" in name:
            short_name = "MiniROI No KF\n(Static + Mask)"
        elif "No Mask" in name:
            short_name = "MiniROI No Mask\n(KF + Raw)"
        else:
            short_name = "MiniROI Baseline\n(Static + Raw)"

        labels.append(short_name)
        drop_rates.append(float(r["Drop_Rate(%)"]))
        f1_scores.append(float(r["F1(%)"]))
        recalls.append(float(r["Recall(%)"]))
        effective_fps.append(float(r["Effective_FPS"]))
        engine_fps.append(float(r["Engine_FPS"]))

    x = np.arange(len(labels))
    width = 0.45

    # 1. Hardware Frame Drop Rate (%)
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(x, drop_rates, width=width, color=colors, edgecolor='black', linewidth=1.2)
    plt.title("Jetson Nano Real-Time Camera Frame Drop Rate (%)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Camera Frames Dropped (%)", fontsize=12)
    plt.xticks(x, labels, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, drop_rates):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    drop_plot_path = os.path.join(output_dir, "figure_nano_realtime_drop_rate.png")
    plt.savefig(drop_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {drop_plot_path}")

    # Copy to artifact directory if exists
    art_dir = r"C:\Users\User\.gemini\antigravity\brain\dac42c7a-e416-43b0-ae7b-dc7800d2667b"
    if os.path.exists(art_dir):
        import shutil
        shutil.copy(drop_plot_path, os.path.join(art_dir, "figure_nano_realtime_drop_rate.png"))

    # 2. Real-Time F1 & Recall Comparison
    plt.figure(figsize=(11, 6))
    w2 = 0.35
    b1 = plt.bar(x - w2/2, recalls, width=w2, label='Real-Time Recall (%)', color='#ff7f0e', edgecolor='black')
    b2 = plt.bar(x + w2/2, f1_scores, width=w2, label='Real-Time F1-Score (%)', color='#2ca02c', edgecolor='black')
    plt.title("Jetson Nano Real-Time Tracking Accuracy Under Frame Drops", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Score (%)", fontsize=12)
    plt.xticks(x, labels, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5)
    for bar in b2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    f1_plot_path = os.path.join(output_dir, "figure_nano_realtime_f1.png")
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {f1_plot_path}")

    if os.path.exists(art_dir):
        import shutil
        shutil.copy(f1_plot_path, os.path.join(art_dir, "figure_nano_realtime_f1.png"))

if __name__ == "__main__":
    c_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study\nano_realtime_summary.csv"
    if not os.path.exists(c_path):
        c_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study\nano_realtime_ablation_summary.csv"
    o_dir = r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study"
    plot_realtime(c_path, o_dir)
