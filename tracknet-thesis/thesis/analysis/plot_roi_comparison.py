
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("[INFO] 載入最終批次盲測實驗數據...")
    # 確保檔名與你實際拿到的完全一致
    df_128 = pd.read_csv("hyperparameter_tuning_results_128_batch.csv")
    df_64 = pd.read_csv("hyperparameter_tuning_results_64_4D.csv")
    
    # 清理欄位名稱可能的空白
    df_128.columns = df_128.columns.str.strip()
    df_64.columns = df_64.columns.str.strip()

    # ==========================================
    # 🏆 定義兩者的「最佳參數組合」
    # ==========================================
    # 64x64 鎖定我們決定的算力極致型: Motion=15, Conf_L=0.55, Conf_G=0.55, Miss=3
    cond_64 = (df_64['Motion_Thresh']==15) & (df_64['Conf_Local']==0.55) & (df_64['Conf_Global']==0.55) & (df_64['Max_Miss']==3)
    best_64 = df_64[cond_64].iloc[0]

    # 128x128 預設鎖定: Motion=15, Conf_L=0.55, Conf_G=0.62, Miss=2 (你可以依據 Excel 實際數據自行修改這四個數字)
    cond_128 = (df_128['Motion_Thresh']==15) & (df_128['Conf_Local']==0.55) & (df_128['Conf_Global']==0.62) & (df_128['Max_Miss']==2)
    if not df_128[cond_128].empty:
        best_128 = df_128[cond_128].iloc[0]
    else:
        # 萬一沒找到，自動挑選 Precision 最高的那組保底
        best_128 = df_128.sort_values(by='Precision(%)', ascending=False).iloc[0]
        print(f"[警告] 找不到預設的 128 參數，已自動切換為 Precision 最高的組合。")

    print(f"✅ 64x64 最佳 FPS: {best_64['Average_FPS']:.2f}, Precision: {best_64['Precision(%)']:.2f}%")
    print(f"✅ 128x128 最佳 FPS: {best_128['Average_FPS']:.2f}, Precision: {best_128['Precision(%)']:.2f}%")

    # ==========================================
    # 圖表一：效能邊界散佈圖 (Pareto Front)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df_128['Average_FPS'], df_128['Precision(%)'], c='royalblue', alpha=0.5, s=60, label='128x128 Engine')
    ax1.scatter(df_64['Average_FPS'], df_64['Precision(%)'], c='crimson', alpha=0.5, s=60, marker='^', label='64x64 Engine')
    
    # 標示黃金點
    ax1.scatter(best_128['Average_FPS'], best_128['Precision(%)'], c='none', edgecolor='black', linewidths=2, s=200, zorder=5)
    ax1.scatter(best_64['Average_FPS'], best_64['Precision(%)'], c='none', edgecolor='black', linewidths=2, s=200, zorder=5)
    ax1.annotate('Best 128x128', (best_128['Average_FPS'], best_128['Precision(%)']), xytext=(-40, -30), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ax1.annotate('Best 64x64', (best_64['Average_FPS'], best_64['Precision(%)']), xytext=(20, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    ax1.set_title("Performance Boundary: Inference Speed vs. Precision (Batch Generalization)", fontsize=14, pad=15)
    ax1.set_xlabel("Average FPS", fontsize=12)
    ax1.set_ylabel("Precision (%)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    fig1.savefig("figure1_pareto_front_batch.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # 圖表二：控制變因折線圖 (64x64 盲推容錯分析)
    # ==========================================
    df_line = df_64[(df_64['Motion_Thresh'] == 15) & (df_64['Conf_Global'] == 0.62)]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors, markers = ['green', 'darkorange', 'purple'], ['o', 's', 'D']
    
    for i, miss in enumerate([1, 2, 3]):
        subset = df_line[df_line['Max_Miss'] == miss].sort_values('Conf_Local')
        ax2.plot(subset['Conf_Local'], subset['Precision(%)'], marker=markers[i], color=colors[i], linewidth=2.5, markersize=8, label=f'Max Miss = {miss}')
        
    ax2.set_title("Sensitivity Analysis: Local Confidence vs. Max Miss (64x64 Batch Data)", fontsize=14, pad=15)
    ax2.set_xlabel("Local Confidence Threshold (Conf_Local)", fontsize=12)
    ax2.set_ylabel("Precision (%)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(title="Tolerance Level")
    fig2.savefig("figure2_sensitivity_line_batch.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # 圖表三：終極對決群組長條圖
    # ==========================================
    labels = ['Precision (%)', 'Recall (%)', 'Local Tracking Ratio (%)']
    val_128 = [best_128['Precision(%)'], best_128['Recall(%)'], best_128['Local_Ratio(%)']]
    val_64 = [best_64['Precision(%)'], best_64['Recall(%)'], best_64['Local_Ratio(%)']]
    
    x = np.arange(len(labels))
    width = 0.35
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    bars1 = ax3.bar(x - width/2, val_128, width, label='Best 128x128', color='royalblue')
    bars2 = ax3.bar(x + width/2, val_64, width, label='Best 64x64', color='crimson')
    
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Ablation Study: Best Configuration on Full Dataset', fontsize=14, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.set_ylim(0, 100) # 展開 Y 軸至 100 以符合完整比例
    ax3.legend(loc='upper left')
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    ax3_twin = ax3.twinx()
    fps_128, fps_64 = best_128['Average_FPS'], best_64['Average_FPS']
    
    ax3_twin.plot([-0.2, 0.2], [fps_128, fps_128], color='darkblue', marker='o', markersize=10, linestyle='None', label='128x128 FPS')
    ax3_twin.plot([2.8, 3.2], [fps_64, fps_64], color='darkred', marker='^', markersize=10, linestyle='None', label='64x64 FPS')
    ax3_twin.plot([0, 3], [fps_128, fps_64], color='black', linestyle=':', alpha=0.3)
    ax3_twin.text(0, fps_128 + 0.5, f'FPS: {fps_128:.1f}', ha='center', va='bottom', color='darkblue', fontweight='bold')
    ax3_twin.text(3, fps_64 + 0.5, f'FPS: {fps_64:.1f}', ha='center', va='bottom', color='darkred', fontweight='bold')

    ax3_twin.set_ylabel('Average FPS (Pure Inference)', fontsize=12)
    ax3_twin.set_ylim(0, max(fps_128, fps_64) + 15) 
    
    fig3.savefig("figure3_ablation_summary_batch.png", dpi=300, bbox_inches='tight')
    
    print("\n🏆 三張終極論文圖表已成功產出！")
    
    # 在 PC 上直接顯示圖表供你預覽
    plt.show()

if __name__ == "__main__":
    main()