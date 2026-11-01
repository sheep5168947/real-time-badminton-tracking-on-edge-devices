# -*- coding: utf-8 -*-
"""
Comprehensive Plotting Suite for all Jetson Nano Hardware Experiments:
1. 128x128 Tournament (match9)
2. 64x64 Tournament (match9)
3. 128x128 Amateur (am_singles)
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

def read_summary(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def plot_all(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    p1_128 = read_summary(os.path.join(output_dir, "nano_ablation_match9_summary.csv"))
    p1_64 = read_summary(os.path.join(output_dir, "nano_ablation_64_match9_summary.csv"))
    am_128 = read_summary(os.path.join(output_dir, "nano_ablation_128_amateur_summary.csv"))
    
    labels = ["Proposed\n(KF+Mask)", "No KF\n(Static+Mask)", "No Mask\n(KF+Raw)", "Baseline\n(Static+Raw)"]
    x = np.arange(len(labels))
    width = 0.35
    
    # -------------------------------------------------------------
    # Plot 1: 128x128 vs 64x64 Throughput Comparison on Jetson Nano
    # -------------------------------------------------------------
    if p1_128 and p1_64:
        fps_128 = [float(r["Average_FPS"]) for r in p1_128]
        fps_64 = [float(r["Average_FPS"]) for r in p1_64]
        
        plt.figure(figsize=(10, 6))
        b1 = plt.bar(x - width/2, fps_128, width, label='128x128 Engine (Standard)', color='#1f77b4', edgecolor='black')
        b2 = plt.bar(x + width/2, fps_64, width, label='64x64 Engine (Ultra-Lightweight)', color='#2ca02c', edgecolor='black')
        
        plt.title("Jetson Nano Real Hardware Throughput: 128x128 vs 64x64 (TensorRT)", fontsize=13, fontweight='bold', pad=15)
        plt.ylabel("Inference FPS", fontsize=11)
        plt.xticks(x, labels, fontsize=10)
        plt.ylim(0, max(max(fps_64), max(fps_128)) * 1.25)
        plt.axhline(30.0, color='red', linestyle='--', linewidth=1.5, label='Real-time Threshold (30 FPS)')
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in b1:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}\nFPS", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in b2:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}\nFPS", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1b5e20')
            
        plt.tight_layout()
        p1_path = os.path.join(output_dir, "figure_nano_128_vs_64_fps.png")
        plt.savefig(p1_path, dpi=300)
        plt.close()
        print(f"[PLOT] Saved: {p1_path}")

    # -------------------------------------------------------------
    # Plot 2: Engine Trigger Distribution (Local % vs Global %)
    # -------------------------------------------------------------
    if p1_128 and am_128:
        loc_tourney = [float(r["Local_Ratio(%)"]) for r in p1_128]
        loc_amateur = [float(r["Local_Ratio(%)"]) for r in am_128]
        
        plt.figure(figsize=(10, 6))
        b3 = plt.bar(x - width/2, loc_tourney, width, label='Tournament Part1 (match9)', color='#2ca02c', edgecolor='black')
        b4 = plt.bar(x + width/2, loc_amateur, width, label='Amateur Singles (am_singles)', color='#ff7f0e', edgecolor='black')
        
        plt.title("Jetson Nano 128x128 Local Search Ratio: Tournament vs Amateur", fontsize=13, fontweight='bold', pad=15)
        plt.ylabel("Local Search Share (%)", fontsize=11)
        plt.xticks(x, labels, fontsize=10)
        plt.ylim(0, 105)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in b3:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in b4:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        p2_path = os.path.join(output_dir, "figure_nano_local_ratio_comparison.png")
        plt.savefig(p2_path, dpi=300)
        plt.close()
        print(f"[PLOT] Saved: {p2_path}")

    # -------------------------------------------------------------
    # Plot 3: F1-Score Across Environments (Tournament 128, 64, Amateur)
    # -------------------------------------------------------------
    if p1_128 and p1_64 and am_128:
        f1_128 = [float(r["F1(%)"]) for r in p1_128]
        f1_64 = [float(r["F1(%)"]) for r in p1_64]
        f1_am = [float(r["F1(%)"]) for r in am_128]
        
        plt.figure(figsize=(11, 6))
        w3 = 0.25
        b5 = plt.bar(x - w3, f1_128, w3, label='Tournament 128x128', color='#1f77b4', edgecolor='black')
        b6 = plt.bar(x, f1_64, w3, label='Tournament 64x64', color='#2ca02c', edgecolor='black')
        b7 = plt.bar(x + w3, f1_am, w3, label='Amateur 128x128', color='#ff7f0e', edgecolor='black')
        
        plt.title("Jetson Nano Tracking Accuracy (F1-Score Comparison)", fontsize=13, fontweight='bold', pad=15)
        plt.ylabel("F1-Score (%)", fontsize=11)
        plt.xticks(x, labels, fontsize=10)
        plt.ylim(0, 100)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bars in [b5, b6, b7]:
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                
        plt.tight_layout()
        p3_path = os.path.join(output_dir, "figure_nano_f1_comparison.png")
        plt.savefig(p3_path, dpi=300)
        plt.close()
        print(f"[PLOT] Saved: {p3_path}")

if __name__ == "__main__":
    out_dir = r"E:\Thesis\tracknet-series-pytorch\report_materials\ablation_study"
    plot_all(out_dir)
