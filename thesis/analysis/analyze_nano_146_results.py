import csv
import numpy as np
import pandas as pd

csv_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\nano_realtime_full_146videos_results.csv"
df = pd.read_csv(csv_path)

total_clips = len(df)
total_frames = df["Total_Frames"].sum()
total_tp = df["TP"].sum()
total_fp = df["FP"].sum()
total_fn = df["FN"].sum()

agg_prec = total_tp / (total_tp + total_fp) * 100.0
agg_rec = total_tp / (total_tp + total_fn) * 100.0
agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec)

mean_prec = df["Precision_Pct"].mean()
median_prec = df["Precision_Pct"].median()

mean_rec = df["Recall_Pct"].mean()
median_rec = df["Recall_Pct"].median()

mean_f1 = df["F1_Score"].mean()
median_f1 = df["F1_Score"].median()

mean_fps = df["FPS"].mean()
median_fps = df["FPS"].median()

mean_drop = df["Drop_Rate_Pct"].mean()
median_drop = df["Drop_Rate_Pct"].median()
zero_drop_clips = (df["Drop_Rate_Pct"] == 0.0).sum()

mean_local = df["Local_Lat_ms"].mean()
mean_rescue = df["Rescue_Lat_ms"].mean()

print("="*80)
print(f"   JETSON NANO REAL-TIME BENCHMARK: FULL 146 VIDEOS MASTER SUMMARY")
print("="*80)
print(f"Total Videos: {total_clips} | Total Frames: {total_frames:,}")
print(f"Total TP: {total_tp:,} | Total FP: {total_fp:,} | Total FN: {total_fn:,}")
print("-" * 80)
print(f"Real-Time FPS:       Mean = {mean_fps:.2f} FPS  | Median = {median_fps:.2f} FPS")
print(f"Hardware Drop Rate:  Mean = {mean_drop:.2f}%     | Median = {median_drop:.2f}%")
print(f"Zero-Drop Clips:     {zero_drop_clips} / {total_clips} ({zero_drop_clips/total_clips*100:.1f}%)")
print(f"Local Latency (ms):  Mean = {mean_local:.2f} ms")
print(f"Rescue Latency (ms): Mean = {mean_rescue:.2f} ms")
print("-" * 80)
print(f"Precision:           Mean = {mean_prec:.2f}%    | Median = {median_prec:.2f}%    | Agg = {agg_prec:.2f}%")
print(f"Recall:              Mean = {mean_rec:.2f}%    | Median = {median_rec:.2f}%    | Agg = {agg_rec:.2f}%")
print(f"F1-Score:            Mean = {mean_f1:.2f}     | Median = {median_f1:.2f}     | Agg = {agg_f1:.2f}")
print("="*80)

# Match by match summary
print("\n" + "="*80)
print("MATCH-BY-MATCH BREAKDOWN (match1 to match23):")
print(f"{'Match':<10} | {'Clips':<5} | {'FPS':<6} | {'Drop%':<7} | {'Prec%':<7} | {'Rec%':<7} | {'F1':<6}")
print("-" * 80)

match_groups = df.groupby("Match", sort=False)
match_summary = []

for m_name, grp in match_groups:
    m_clips = len(grp)
    m_fps = grp["FPS"].mean()
    m_drop = grp["Drop_Rate_Pct"].mean()
    m_prec = grp["Precision_Pct"].mean()
    m_rec = grp["Recall_Pct"].mean()
    m_f1 = grp["F1_Score"].mean()
    print(f"{m_name:<10} | {m_clips:<5} | {m_fps:>5.1f}  | {m_drop:>5.1f}%  | {m_prec:>5.1f}% | {m_rec:>5.1f}% | {m_f1:>5.1f}")
    match_summary.append({
        "Match": m_name, "Clips": m_clips, "FPS": m_fps, "Drop_Rate": m_drop,
        "Precision": m_prec, "Recall": m_rec, "F1": m_f1
    })

match_summary_df = pd.DataFrame(match_summary)
out_match_summary = r"E:\Thesis\tracknet-series-pytorch\report_materials\nano_match_by_match_summary.csv"
match_summary_df.to_csv(out_match_summary, index=False)
print(f"\nMatch summary saved to {out_match_summary}")
