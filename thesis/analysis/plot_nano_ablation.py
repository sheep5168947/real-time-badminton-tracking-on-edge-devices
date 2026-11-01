# -*- coding: utf-8 -*-
"""
Generate Visualization Plots for Jetson Nano Hardware Ablation Results.
Plots:
1. Jetson Nano Average FPS & Per-Frame Latency (Proposed vs No KF vs No Mask vs Baseline)
2. Local Search Ratio vs Global Recovery Ratio
3. Tracking Accuracy (Precision, Recall, F1)
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for server/script execution
plt.switch_backend('Agg')

def generate_nano_ablation_plots(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labels = []
    fps_list = []
    local_ratios = []
    global_ratios = []
    precisions = []
    recalls = []
    f1s = []

    for r in rows:
        kf = r.get("KF", "")
        mask = r.get("Motion_Mask", "")
        if kf == "ON" and mask == "ON":
            name = "Proposed\n(KF + Mask)"
        elif kf == "OFF" and mask == "ON":
            name = "No KF\n(Static + Mask)"
        elif kf == "ON" and mask == "OFF":
            name = "No Mask\n(KF + Raw)"
        else:
            name = "Baseline\n(Static + Raw)"

        labels.append(name)
        fps = float(r.get("Average_FPS", 0))
        fps_list.append(fps)
        local_ratios.append(float(r.get("Local_Ratio(%)", 0)))
        global_ratios.append(float(r.get("Global_Ratio(%)", 0)))
        precisions.append(float(r.get("Precision(%)", 0)))
        recalls.append(float(r.get("Recall(%)", 0)))
        f1s.append(float(r.get("F1(%)", 0)))

    x = np.arange(len(labels))
    width = 0.5

    # 1. Jetson Nano Hardware FPS Plot
    plt.figure(figsize=(9, 6))
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#7f7f7f']
    bars = plt.bar(x, fps_list, width=0.45, color=colors, edgecolor='black', linewidth=1.2)
    plt.title("Jetson Nano Real Hardware Throughput (TensorRT)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Average Inference FPS", fontsize=12)
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0, max(fps_list) * 1.25 if fps_list else 40)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, fps in zip(bars, fps_list):
        yval = bar.get_height()
        ms = round(1000.0 / fps, 1) if fps > 0 else 0
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{fps:.1f} FPS\n({ms} ms)",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    fps_plot_path = os.path.join(output_dir, "figure_nano_fps_comparison.png")
    plt.savefig(fps_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {fps_plot_path}")

    # 2. Local vs Global Ratio Plot (Stacked)
    plt.figure(figsize=(9, 6))
    p1 = plt.bar(x, local_ratios, width=0.45, color='#2ca02c', label='128x128 Local Search (Fast ~33ms)')
    p2 = plt.bar(x, global_ratios, width=0.45, bottom=local_ratios, color='#ff7f0e', label='512x288 Global Recovery (Slow ~210ms)')
    plt.title("Jetson Nano Engine Trigger Ratio (Efficiency Impact)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Execution Ratio (%)", fontsize=12)
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0, 115)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(labels)):
        loc = local_ratios[i]
        glo = global_ratios[i]
        if loc > 5:
            plt.text(x[i], loc / 2, f"{loc:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        if glo > 5:
            plt.text(x[i], loc + glo / 2, f"{glo:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=11)

    plt.tight_layout()
    ratio_plot_path = os.path.join(output_dir, "figure_nano_engine_ratio.png")
    plt.savefig(ratio_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {ratio_plot_path}")

    # 3. Accuracy Metrics (Precision, Recall, F1)
    plt.figure(figsize=(10, 6))
    w = 0.25
    plt.bar(x - w, precisions, width=w, label='Precision', color='#1f77b4')
    plt.bar(x, recalls, width=w, label='Recall', color='#ff7f0e')
    plt.bar(x + w, f1s, width=w, label='F1-Score', color='#2ca02c')
    plt.title("Jetson Nano Tracking Accuracy (match9)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Score (%)", fontsize=12)
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0, 105)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(labels)):
        plt.text(x[i] - w, precisions[i] + 1.5, f"{precisions[i]:.1f}", ha='center', fontsize=9)
        plt.text(x[i], recalls[i] + 1.5, f"{recalls[i]:.1f}", ha='center', fontsize=9)
        plt.text(x[i] + w, f1s[i] + 1.5, f"{f1s[i]:.1f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    acc_plot_path = os.path.join(output_dir, "figure_nano_accuracy_comparison.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {acc_plot_path}")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study\nano_ablation_match9_summary.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study"
    generate_nano_ablation_plots(csv_file, out_dir)
