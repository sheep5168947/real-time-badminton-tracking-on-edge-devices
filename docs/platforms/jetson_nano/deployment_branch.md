# Nano 部署分支

此分支從正式 main 建立，保留遷移前 Nano 上已存在的 Python 3.6 相容性原碼差異。
不是重新設計模型或修改追蹤機制，也不以新版 PC PyTorch 套件替換裝置環境。
原本的 Nano 小型實驗 CSV 與 log 也保留在此部署分支，避免被 PC 同名結果覆寫。

在 Nano 從 deploy/jetson-nano 開工作分支，完成裝置端相關檢查、commit、push，PR 回此部署分支。
通用功能先在 main 開發，再用部署整合工作分支進行相容性驗證。
運行入口仍是 thesis/platforms/jetson_nano/，raw/ 和 weights/ 留在本機。
PC 的 requirements.txt 只作來源紀錄，Nano 不應執行 pip install -r requirements.txt。
