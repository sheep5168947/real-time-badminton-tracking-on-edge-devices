# 從初始快照轉為正式版控

Repository: https://github.com/sheep5168947/real-time-badminton-tracking-on-edge-devices

專案標題依 PPT 設為 Real-Time Badminton Tracking on Edge Devices: A Lightweight ROI-based Architecture。
原 repo 名稱 bottle-defect-detection-aoi 已改名；原先快照分支只留作歷史。
正式原碼移到 repo 根目錄，不再維護 tracknet-thesis 備份子目錄。

main 是 PC／通用開發基底，deploy/jetson-nano 是有實際相容性差異的部署基底。
舊上游 Git metadata 在轉換前保存在本機封存；來源 README、Nano 相容差異及原碼封存仍留在 docs/。
遷移不修改追蹤模型、熱力圖、mask、KF 或既有訓練參數。

本機 hooks 為每個 clone 設定，不會自動跨機器啟用；請執行 python scripts/install_git_hooks.py。
大型資產留在工作機，Git fetch、切分支或建立版控不應刪除它們。
