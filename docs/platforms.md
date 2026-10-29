# PC 與 Jetson Nano 平台分類

```text
thesis/platforms/
  pc/                         PyTorch 即時推論
    teacher_realtime.py
    student_realtime_events.py
    student_realtime_legacy.py
  jetson_nano/                 保留 Nano 原始實作及預設參數
    runtime/                  教師、128/64 學生、GT 展示
    evaluation/               批次、容忍距離、參數掃描
    experiments/              早期純 ROI 展示
    diagnostics/              GStreamer／ONNX 診斷，舊 PyTorch 測速
```

## PC

在專案根目錄、tracknet 環境執行：

```powershell
python -m thesis.platforms.pc.student_realtime_events --video "影片.mp4"
python -m thesis.platforms.pc.teacher_realtime
```

資料與 checkpoint 位置未改，仍使用 PC 原本的 dataset/、outputs/、checkpoints*/。

## Jetson Nano

在 Nano 專案根目錄執行，不要直接匯入 PC 程式：

```bash
python3 -m thesis.platforms.jetson_nano.runtime.student_128 --motion 10 --local_conf 0.55 --global_conf 0.55 --miss 3
python3 -m thesis.platforms.jetson_nano.runtime.student_64 --motion 10 --local_conf 0.55 --global_conf 0.55 --miss 3
python3 -m thesis.platforms.jetson_nano.runtime.teacher_v4
python3 -m thesis.platforms.jetson_nano.runtime.demo_with_gt --roi_size 128 --input raw/part1/video/example.mp4 --csv raw/part1/csv/example_ball.csv --output demo.mp4
```

影片範例路徑請換成實際存在的檔案。weights/ 的 engines、影片與 GT 未搬移。
Nano 原始預設參數與 PC 事件版不完全相同，以上用明確參數示範，不替你改動歷史預設值。
目前偵測到 Nano 的系統 Python 為 3.6.9；保留其既有 TensorRT／PyCUDA 環境，沒有安裝 PC requirements。
若另有 tracknet 環境，先核對它的 Python 與 TensorRT 可用性，再按 AGENTS.md 選環境。
這輪只驗證語法、路徑與純命令列入口，不宣稱已測得新的 Nano 推論 FPS／精度。

## Nano 評估與掃描

```bash
python3 -m thesis.platforms.jetson_nano.evaluation.batch_evaluate_128 --motion 10 --local_conf 0.55 --global_conf 0.55 --miss 3
python3 -m thesis.platforms.jetson_nano.evaluation.evaluate_tolerances_128
python3 -m thesis.platforms.jetson_nano.evaluation.evaluate_tolerances_64
python3 -m thesis.platforms.jetson_nano.evaluation.evaluate_tolerances_32
python3 -m thesis.platforms.jetson_nano.evaluation.sweep_128
python3 -m thesis.platforms.jetson_nano.evaluation.sweep_64
```

掃描會執行 81 組評估，請勿只為測試搬移而啟動。啟動 GPU 工作前依 AGENTS.md 檢查現有程序。
批次評估仍讀取 Nano 的 raw/part1，輸出 CSV 名稱與原本一致。兩支 sweep 的子程序路徑已改成新檔案的實際位置。
這些 Nano 歷史評估的 sigmoid、GT 時間對齊與原始像素容忍度尚未全面稽核，不可直接和新的 PC 配對消融分數混用。

## 來源與備份

Nano 原始程式與既有上游修改先完成來源快照。PC 保留 Nano 原碼，Nano 則只安裝其平台原碼與說明，不覆蓋既有 tracknet/。
`docs/platforms/jetson_nano/upstream_local_changes.patch` 記錄整理前 Nano 上游原碼相對 PC 的差異，僅供備份追溯；不要在 PC 盲目套用。
改名清單見 `docs/platforms/jetson_nano/file_migration.csv`。
舊檔名 `realtime_student_nano_64*64.py` 已換成 `runtime/student_64.py`，可在 Windows 正常保存。

## Git 與部署分支

專案已改為正式開發 repo，工作目錄即為 repo 根目錄。
PC 使用 main，Nano 使用 deploy/jetson-nano 保存既有 Python 3.6 相容性差異。
每次從對應基底開 task branch，檢查後 commit、push，透過 PR 合併。詳見 CONTRIBUTING.md。
commit 作者為 Nero；憑證不寫入 repo。原始影片、raw/、weights/、ONNX／TensorRT engines、密碼及暫存輸出不進一般 Git。
Git 原碼版控不等同資料集與 engines 的完整備份。
