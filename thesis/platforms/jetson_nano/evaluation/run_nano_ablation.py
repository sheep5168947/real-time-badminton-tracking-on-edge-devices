# -*- coding: utf-8 -*-
"""
Jetson Nano Automated Ablation Runner
Runs 4 ablation variants sequentially on Jetson Nano using TensorRT:
1. Proposed: Full (KF + Motion Mask)
2. No KF: Static ROI + Motion Mask
3. No Mask: KF + Raw Heatmap
4. Baseline: Static ROI + Raw Heatmap

Compatible with Python 3.6+ and TensorRT on Jetson Nano.
"""

import os
import sys

# Critical runtime guards for Jetson Nano (Cortex-A57 & OpenMP TLS)
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
if "/usr/lib/aarch64-linux-gnu/libgomp.so.1" not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1:" + os.environ.get("LD_PRELOAD", "")

import argparse
import subprocess
import csv
import json
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run 4-Variant Ablation Study on Jetson Nano")
    parser.add_argument("--data_dir", type=str, default="raw/part1", help="Dataset directory containing video and csv subfolders")
    parser.add_argument("--output_dir", type=str, default="report_materials/nano_ablation", help="Directory to save ablation results")
    parser.add_argument("--roi_size", type=int, default=128, choices=[64, 128], help="ROI Size (64 or 128, default: 128)")
    parser.add_argument("--motion", type=int, default=10, help="Motion Mask Threshold (default: 10)")
    parser.add_argument("--local_conf", type=float, default=0.55, help="Local Confidence Threshold (default: 0.55)")
    parser.add_argument("--global_conf", type=float, default=0.55, help="Global Confidence Threshold (default: 0.55)")
    parser.add_argument("--miss", type=int, default=3, help="Max Miss Tolerance (default: 3)")
    return parser.parse_args()

VARIANTS = [
    {
        "id": "proposed",
        "name": "Full (Proposed: KF + Mask)",
        "extra_args": []
    },
    {
        "id": "no_kf",
        "name": "No KF (Static ROI + Mask)",
        "extra_args": ["--no_kf"]
    },
    {
        "id": "no_mask",
        "name": "No Mask (KF + Raw Heatmap)",
        "extra_args": ["--no_mask"]
    },
    {
        "id": "baseline",
        "name": "Baseline (Static ROI + Raw)",
        "extra_args": ["--no_kf", "--no_mask"]
    }
]

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv_path = os.path.join(args.output_dir, "nano_ablation_summary.csv")
    
    # Clean previous summary if exists
    if os.path.exists(summary_csv_path):
        os.remove(summary_csv_path)
        
    print("================================================================================")
    print("[START] Jetson Nano 4-Variant 消融實驗啟動")
    print("資料集路徑: {}".format(args.data_dir))
    print("超參數: Motion Thresh={}, Local Conf={}, Global Conf={}, Max Miss={}".format(
        args.motion, args.local_conf, args.global_conf, args.miss
    ))
    print("結果輸出目錄: {}".format(args.output_dir))
    print("================================================================================\n")
    
    results = []
    
    for idx, var in enumerate(VARIANTS, 1):
        print("--------------------------------------------------------------------------------")
        print("[{}/4] 正在評估變體: {}".format(idx, var["name"]))
        print("--------------------------------------------------------------------------------")
        
        eval_module = "thesis.platforms.jetson_nano.evaluation.batch_evaluate_{}".format(args.roi_size)
        cmd = [
            sys.executable, "-m", eval_module,
            "--base_dir", args.data_dir,
            "--motion", str(args.motion),
            "--local_conf", str(args.local_conf),
            "--global_conf", str(args.global_conf),
            "--miss", str(args.miss),
            "--output_csv", summary_csv_path
        ] + var["extra_args"]
        
        start_t = time.time()
        ret = subprocess.call(cmd)
        elapsed = time.time() - start_t
        
        if ret != 0:
            print("[警告] 變體 {} 執行失敗 (Exit Code: {})".format(var["name"], ret))
        else:
            print("[完成] 變體 {} 完成，耗時: {:.1f}s\n".format(var["name"], elapsed))
            
    # Read back summary CSV and display pretty table
    if os.path.exists(summary_csv_path):
        print("\n================================================================================")
        print("[SUMMARY] Jetson Nano 消融實驗完整對照表")
        print("================================================================================")
        with open(summary_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        print("{:<28} | {:<7} | {:<7} | {:<12} | {:<12} | {:<9} | {:<9} | {:<9}".format(
            "變體名稱 (Variant)", "KF", "Mask", "Local(%)", "Global(%)", "FPS", "Precision", "F1(%)"
        ))
        print("-" * 105)
        for row in rows:
            kf = row.get("KF", "")
            mask = row.get("Motion_Mask", "")
            name = "Proposed (KF+Mask)" if kf=="ON" and mask=="ON" else \
                   "No KF (Static+Mask)" if kf=="OFF" and mask=="ON" else \
                   "No Mask (KF+Raw)" if kf=="ON" and mask=="OFF" else "Baseline (Static+Raw)"
            print("{:<28} | {:<7} | {:<7} | {:<12} | {:<12} | {:<9} | {:<9} | {:<9}".format(
                name, kf, mask,
                row.get("Local_Ratio(%)", ""),
                row.get("Global_Ratio(%)", ""),
                row.get("Average_FPS", ""),
                row.get("Precision(%)", "") + "%",
                row.get("F1(%)", "") + "%"
            ))
        print("================================================================================")
        print("詳細 CSV 已儲存至: {}".format(summary_csv_path))
    else:
        print("[錯誤] 未找到生成的彙總 CSV: {}".format(summary_csv_path))

if __name__ == "__main__":
    main()
