# PPT 流程與執行入口

所有指令均在專案根目錄、tracknet 環境執行。訓練與評估需先檢查 GPU，依 AGENTS.md 的 GPU 與程序規範操作。
此輪沒有啟動訓練、匯出模型、移動資料或更換權重。程式搬移不代表原實驗結論已驗證。

|階段|入口|資料與用途|
|---|---|---|
|教師資料前處理|`python -m tracknet.tools.preprocess --config configs/preprocess_fast_512x288.yaml`|沿用上游管線；先核對 splits 與資料根目錄|
|教師 V4 訓練|`python -m tracknet.tools.train --config configs/train_v4.yaml`|保留原配置；多 GPU 啟動方式遵循 AGENTS.md|
|教師 V4 評估|`python -m tracknet.tools.evaluate --config configs/evaluate_v4.yaml`|完整管線見 docs/upstream_readme.md|
|原始目錄整理|`python -m thesis.data.organize_raw_dataset`|會搬移 raw 下 match 目錄，只在確實需要時執行|
|蒸餾快取|`python -m thesis.data.cache_teacher_heatmaps`|part1 → cached_pt_v4；會產生／覆寫快取|
|128 學生訓練|`python -m thesis.training.train_student_128`|保留加權損失及 checkpoints/epoch_30.pth 存在時的初始化邏輯|
|64 學生訓練|`python -m thesis.training.train_student_64`|由頭訓練；保存至 checkpoints_v4_student_64x64|
|32 學生訓練|`python -m thesis.training.train_student_32`|早期壓縮試驗保留；不是目前 PPT 主線預設|
|教師匯出|`python -m thesis.export.export_teacher_v4`|原教師 checkpoint → tracknet_v4_512x288.onnx|
|128 局部匯出|`python -m thesis.export.export_student_128`|128 學生權重 → miniroi_128x128.onnx|
|512 全域匯出|`python -m thesis.export.export_student_global`|相同 128 學生權重 → miniroi_512x288.onnx|
|64 匯出|`python -m thesis.export.export_student_64`|64 學生權重 → student_model_64x64.onnx|
|32 匯出|`python -m thesis.export.export_student_32`|32 學生權重 → weights/miniroi_32x32.onnx|
|教師 PC 推論|`python -m thesis.platforms.pc.teacher_realtime`|保留原影片及權重路徑|
|學生 PC 推論及事件|`python -m thesis.platforms.pc.student_realtime_events --video <影片>`|目前用來核對行為的即時版，預設 mask=10/conf=0.55/miss=3|
|早期學生 PC 展示|`python -m thesis.platforms.pc.student_realtime_legacy`|保留歷史版本，設定不等同目前事件版|
|mask 候選搜尋|`python -m thesis.analysis.find_mask_cases --video <影片> --mode offline`|候選不等同 GT 證實的成功案例|
|ROI 對照繪圖|`python -m thesis.analysis.plot_roi_comparison`|讀取根目錄兩份 hyperparameter_tuning_results CSV|

教師 checkpoint 的既有路徑為 `outputs/train/tracknet_v4_20260715_075947/checkpoints/best.pt`。
現階段保留來源不改，但正式論文應核對完成訓練 checkpoint 的使用政策。
學生 epochs、batch size、資料來源及權重輸出目錄均未修改。

## 有／無 mask 比較

```powershell
python -m thesis.evaluation.compare_motion_mask --manifest configs/thesis/evaluation_original58.csv
python -m thesis.evaluation.compare_motion_mask --manifest configs/thesis/evaluation_amateur7.csv --output-prefix amateur_mask_ablation
```

清單的路徑以專案根目錄為基準；輸出至 model_results/evaluation/<時間戳>/。
兩版本有獨立追蹤狀態，GT 不參與推論。記錄中間／最新幀、模型偵測／KF 畫面位置、15/20/30/50 原圖像素誤差。
無 GT 的影格排除，不能當作無球。清單只是評估來源，不證明資料從未用於教師或學生訓練。

## 歷史消融

`thesis/evaluation/historical/` 保留 local-only、global-only、without-KF、without-mask 的原始實驗版本。
它們的 sigmoid 次數、頂部遮蔽、信心門檻或失追行為可能與目前事件版不同。
只用於追溯 PPT 既有實驗，不可直接將它們彼此比較，聲稱只有單一機制改變。
本次沒有修正這些機制，避免改變歷史結果。

## 已封存的根目錄舊程式

eval_tracker.py、refactored_eval.py、generate_thesis_plots.py、visualize_tracker.py 已被目前事件版、比較程式或最終繪圖版本取代。
原碼已收進本次整理的外部備份，不留在活躍原碼樹。
完整改名對照見 file_migration.csv。
