# Edge Badminton Tracking Thesis

本專案以 TrackNet 系列程式為基礎，研究 Jetson Nano 上的輕量羽球 ROI 追蹤。
目前只整理程式位置、名稱與操作文件，沒有修改追蹤機制或訓練超參數。

## 目錄

```text
thesis/
  models/       MiniROI 學生模型
  data/         原始資料整理、教師熱力圖快取
  training/     128 / 64 / 32 學生訓練
  export/       教師與學生 ONNX 匯出
  platforms/    PC 與 Jetson Nano 的平台專屬入口
  evaluation/   獨立狀態機有／無 mask 評估
    historical/ 舊消融程式，只供實驗追溯
  analysis/     ROI 圖表與 mask 候選搜尋
tracknet/       保留的上游模型、教師訓練、前處理與依賴
configs/thesis/ 可攜式評估影片清單
docs/           操作流程、搬移對照、歷史注意事項
dataset/        原始資料、處理後資料與蒸餾快取（未搬動）
checkpoints*/   已有學生權重（未搬動）
outputs/        已有教師訓練成果（未搬動）
```

## 執行方式

先啟用 `tracknet` 環境，並將工作目錄切到本專案根目錄。
建議使用 `python -m thesis.<分類>.<名稱>`，詳見 [PPT 流程與指令](docs/thesis_workflow.md)。
訓練、匯出與部分舊程式仍保留原本的根目錄相對路徑，執行前請核對資料與權重。

```powershell
conda activate tracknet
python -m thesis.platforms.pc.student_realtime_events --help
python -m thesis.evaluation.compare_motion_mask --self-test
```

## 評估注意事項

mask 的效果尚未獲得整體收益證據。歷史評估與正式比較不可混用。
訓練資料重疊、GT 時間對齊、距離容忍度及是否包含 KF 補位必須分開記錄。
現在的比較使用逐幀輸入，不代表即時跳幀準確度或裝置 FPS。

## 上游來源

整理前 remote： https://github.com/AnInsomniacy/tracknet-series-pytorch.git
這裡將它記為上游來源，不宣稱所有程式均由本論文原創，也不據此認定為作者官方版本。
原 README 保存在 [上游操作說明](docs/upstream_readme.md)。Git remote 與歷史本輪不修改。

## 檢查

```powershell
python -m compileall -q tracknet thesis tests scripts
python -m pytest -q
```

額外開發依賴列於 requirements-dev.txt。平台分類已完成，見 [平台操作說明](docs/platforms.md)。個人遠端 Git repo 另行設定。

平台原碼已分開管理，PC 不需安裝 TensorRT／PyCUDA，Nano 不需執行 PC 的 PyTorch 入口。
