# 開發與推送規則

本專案使用 GitHub Flow 與 Conventional Commits。這是開發版控，不是快照備份機制。

## 分支

- main：通用原碼與 PC 版本，透過 Pull Request 合併，不直接推送。
- deploy/jetson-nano：Nano 部署版本，保留裝置既有 Python 3.6 相容性修改，不是備份分支。
- 每個任務從適當基底建立短期分支，例如 feat/roi-recovery、fix/nano-input、docs/evaluation-protocol。
- 允許 feat/、fix/、docs/、refactor/、perf/、test/、chore/、build/、ci/、style/ 工作分支。
- PC 工作通常從 main 開始；只涉及 Nano 相容性的工作從 deploy/jetson-nano 開始，PR 合併回該部署分支。
- 通用改動先合併 main，再以獨立部署工作分支整合到 Nano，驗證後 PR 回部署分支。
- 不強推、不改寫已發布歷史、不自動 stash 或刪除使用者未提交修改。

## Commit

格式：type(scope): description。scope 可省略，破壞性改動用 ! 或 BREAKING CHANGE 說明。

```text
feat(roi): add recovery state logging
fix(nano): resolve relative engine path
refactor(pc): separate realtime entry points
docs: document evaluation frame alignment
test: verify mask threshold behavior
```

主旨最多 100 字元，需有清楚描述。每個 commit 完成一項相關變動，避免 update、backup、final 等模糊訊息。
作者使用本機 repo 設定的 Nero，不把帳號憑證或 SSH 密碼放進檔案。

## 每次修改流程

```bash
git status
git checkout main
git pull --ff-only
git checkout -b fix/example-topic
# 修改並執行相關檢查
git add <明確的檔案>
git diff --cached
git commit -m "fix(scope): describe the change"
git push -u origin HEAD
```

Nano 將基底改成 deploy/jetson-nano。範例使用 git checkout，兼容裝置上較舊的 Git。未提交修改存在時，先處理它們，不要直接切分支或覆蓋。
推送後開 PR，列出目的、驗證與限制。檢查通過後再合併並更新裝置；不自動要求自己批准自己的 PR。
經使用者授權的工作，助理完成相關修改與驗證後應 commit 並 push 到工作分支，回報分支與 commit；不得只留本機修改。
例外是使用者明確要求暫不提交，或認證／檢查失敗時。此時說明狀態，不得宣稱已推送。

## 檢查

PC 使用 tracknet 環境：

```bash
python -m compileall -q tracknet thesis tests scripts
python -m pytest -q
git diff --check
```

Nano 使用既有裝置環境，只編譯／驗證 Nano 平台程式與相關入口。
不為了版控檢查安裝 PC 的 PyTorch 或更換 TensorRT，也不自動啟動 81 組掃描或真實訓練。
如果 pytest 在該裝置不可用，明確回報實際完成的檢查；不要把 PC 測試當成 Nano 端到端驗證。

## 本機 hooks

```bash
python scripts/install_git_hooks.py
```

commit-msg 驗證新 commit 的 Conventional Commits 格式。
pre-push 阻止直接推 main／部署分支及 backup 工作分支，並解析本次 push 的 Python 改動，不啟動模型或訓練。
Nano 只解析其平台與版控工具，避免以 Python 3.6 誤檢 PC 的新版語法。
hooks 不取代完整測試與 PR。歷史來源的空白格式差異不應混入演算法修改；只檢查本次變動。

## 資料與大型檔案

資料集、weights、checkpoints、ONNX、TensorRT engines、產出影片與密碼不提交。
PPT 已有歷史版本，後續大幅修改時評估 Git LFS，不在此次遷移中重寫歷史。
既有 backup/tracknet-thesis-platforms-20261030 僅保留遷移歷史，不再作日常工作分支。

參考：
- https://docs.github.com/en/get-started/using-github/github-flow
- https://www.conventionalcommits.org/en/v1.0.0/
