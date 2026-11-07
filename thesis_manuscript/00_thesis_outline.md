# 碩士學位論文大綱與撰寫架構規範 (Master's Thesis Outline)

## 論文題目 (Thesis Title)
- **中文題目**：基於階層式微型神經網絡與邊緣運算之即時羽球高速追蹤系統
- **英文題目**：Real-Time High-Speed Badminton Shuttlecock Tracking via Hierarchical Mini-ROI Neural Networks on Edge Embedded Systems

---

## 論文架構總覽 (Table of Contents)

### 誌謝 (Acknowledgements)
### 中文摘要 (Abstract - Traditional Chinese)
### 英文摘要 (Abstract - English)
### 目錄 (Table of Contents)
### 圖目錄 (List of Figures)
### 表目錄 (List of Tables)

---

### 第一章 緒論 (Chapter 1: Introduction)
- **1.1 研究背景與實務需求 (Research Background & Practical Demands)**
  - 羽球運動高速特性（最高球速突破 400 km/h、高速位移與形變、動態模糊）。
  - 現行商業鷹眼（Hawk-Eye）多相機陣列之高昂成本與封閉性。
  - 單視角（Monocular Camera）電腦視覺在體育科技與業餘賽事普及化之重要性。
- **1.2 邊緣計算之困境與研究動機 (Edge Computing Dilemma & Motivation)**
  - 深度學習追蹤模型（TrackNet V1-V5 系列）之全圖卷積運算瓶頸。
  - 邊緣硬體（以 10W NVIDIA Jetson Nano 為例）之算力、頻寬與散熱限制。
  - 「離線評測（Offline）」掩蓋之即時性陷阱：相機 30 FPS 硬體串流攝入下，全域搜索時延（~120ms）引發之排隊溢出與災難性掉幀（Hardware Frame Drops）。
  - 假陽性（False Positive）對落點重建之致命危害（地坪眩光、球員白鞋、球場標線誤判）。
- **1.3 研究目標與核心貢獻 (Research Objectives & Core Contributions)**
  - 貢獻一：**階層式粗細兩階段邊緣追蹤架構（Hierarchical Coarse-to-Fine Edge Tracking）**。
  - 貢獻二：**物理運動學自適應動態門檻（Dynamic Smash Gate & Velocity Compensation）**。
  - 貢獻三：**高純度雙影格動態確認與嚴格微型驗證機制（Precision $\ge 95\%$ 防禦機制）**。
  - 貢獻四：**嚴格硬體掉幀懲罰之全量錦標賽與跨域業餘實機驗證（146 支影片、近 7 萬影格完整實測）**。
- **1.4 論文架構概述 (Thesis Organization)**

---

### 第二章 相關研究與文獻探討 (Chapter 2: Related Work)
- **2.1 高速微小球體視覺追蹤技術 (High-Speed Small Ball Tracking)**
  - 傳統影像處理方法（背景消影、光流法、霍夫變換之局限性）。
  - TrackNet 系列之演進（TrackNet V1 ~ V5 特徵與限制分析）。
  - 多目標追蹤與時序運動引導網絡（Motion-guided Temporal Fusion）。
- **2.2 深度神經網絡輕量化與知識蒸餾 (Model Compression & Knowledge Distillation)**
  - 模型剪枝（Pruning）、量化（Quantization, TensorRT FP16/INT8）。
  - 教師－學生模型架構（Teacher-Student Distillation）。
  - 局域感興趣區域（Region of Interest, ROI）聚焦技術之演進。
- **2.3 邊緣嵌入式平台之即時串流架構 (Real-Time Edge Streaming Architectures)**
  - 雙執行緒生產者－消費者模式（Producer-Consumer Threading）。
  - 硬體掉幀對時序模型狀態機之影響分析。
  - 現存文獻評測缺陷：離線假即時 vs. 實體硬體攝入真即時。

---

### 第三章 階層式邊緣羽球追蹤方法論 (Chapter 3: Methodology)
- **3.1 邊緣端羽球追蹤之基準範式與領域核心挑戰 (Baseline Paradigms & Domain Challenges)**
  - 業界常見基準範式評估（深層全解析度卷積、全畫面均勻空間降採樣、時空運動遮罩與初階階層架構之物理瓶頸）。
  - 邊緣即時羽球追蹤之四大核心挑戰（極速脫靶 Tail-Dragging Lag、反彈過衝 Kinematic Reversal、地坪反光陷阱 Specular Reflection、靜態微震盪 Static Micro-Jitter）。
- **3.2 系統整體架構設計 (System Architecture & Pipeline Overview)**
  - 系統端到端推論管線（End-to-End Inference Pipeline）。
  - 追蹤狀態機導引與狀態轉移（Tracking Finite State Machine）。
  - 零拷貝單影格增量環形緩衝區（Zero-Copy Incremental Ring Buffer）。
- **3.3 模組一：時空適應性局部追蹤機制 (Module 1: Spatio-Temporal Adaptive Local Tracking)**
  - 扣殺搜尋窗動態擴展（Dynamic Smash ROI Expansion）與次像素質心連續微調（Sub-pixel Refinement）。
  - 無此模組 vs. 啟用此模組之視覺現象與量化數據對比（Without vs. With Comparison）。
- **3.4 模組二：擊球動態轉折與反向速度阻尼機制 (Module 2: Kinematic Reversal Damping Mechanism)**
  - 向量內積方向突變檢驗（Dot-Product Inversion Detection）與慣性過衝抑制。
  - 無此模組 vs. 啟用此模組之視覺現象與量化數據對比（Without vs. With Comparison）。
- **3.5 模組三：高純度階層式全域救援機制 (Module 3: High-Purity Hierarchical Global Rescue)**
  - 峰值均值比（PMR）粗篩、原生微型高置信度驗證（$\tau_{\text{verify}} \ge 0.35$）與動態飛行雙影格物理確認（2-Hit Motion Confirmation）。
  - 斬除極端地坪反光（`match10` FP 消除 97.2%）之視覺與全量賽事實測數據。
- **3.6 模組四：狀態機管制與靜止死球自適應凍結 (Module 4: State Machine Regulation & Static Freeze)**
  - 4 幀連續微小位移監測與主動休眠脫鎖機制。
  - 無此模組 vs. 啟用此模組之視覺現象與量化數據對比（Without vs. With Comparison）。
- **3.7 方法論總結：四大模組協同防禦矩陣 (Collaborative Defense Matrix)**

---

### 第四章 嵌入式硬體即時部署與管線優化 (Chapter 4: Edge Implementation & Hardware Optimization)
- **4.1 NVIDIA Jetson Nano 邊緣硬體特性分析**
  - Maxwell 128-core GPU 記憶體架構、頻寬瓶頸與 4GB 共享記憶體限制。
  - 10W MAXN 功耗限制下的熱節流（Thermal Throttling）防範機制。
- **4.2 TensorRT FP16 引擎編譯與鎖頁記憶體最佳化**
  - 動態 Shape 與固定維度引擎之推論延遲對比。
  - Page-locked Host Memory（Pinned Memory）與非同步 CUDA Stream 資料傳輸管線。
- **4.3 嚴格 30 FPS 硬體串流攝入佇列與掉幀懲罰機制**
  - 雙執行緒非阻塞環形佇列（Queue Capacity = 3）。
  - 硬體掉幀即時統計與雙重懲罰標註（Drop-Aware Strict Accounting）。

---

### 第五章 實驗結果與效能評估 (Chapter 5: Experiments & Evaluation)
- **5.1 實驗環境設定與評測指標定義**
  - 嚴格雙重懲罰標準（$\text{tolerance} = 15.0\text{ px}$，偏差過大同時計為 FP 與 FN）。
  - 錦標賽資料集（TrackNet Part 1: 23 場賽事，146 支影片，69,396 影格）。
  - 新加坡未見真實業餘羽球館資料集（Cross-Domain Amateur Dataset）。
- **5.2 全量 146 支影片跨賽事巨量實測分析**
  - 總體聚合指標（FPS, Drop Rate, Precision, Recall, F1）。
  - 23 場錦標賽分場詳盡數據對比（Match-by-Match Breakdown）。
  - 19 場常規標準轉播賽事子集分析（Precision $\ge 96\%$, Recall $\ge 80\%$）。
- **5.3 極端邊界條件場景深度剖析**
  - 超高空俯拍視角（`match6` 台北公開賽，微小球 2px 定位）。
  - 頂棚強光光暈與暴力扣殺（`match8` 探照燈穿透）。
  - 白色地坪強烈鏡面眩光（`match10` 98%+ 假陽性消減實證）。
- **5.4 跨域泛化性實測：新加坡社區球場真實對決**
  - 新加坡業餘賽事實測數據與視覺化軌跡。
  - 與官方 PC 離線 Teacher Model（TrackNetV4）之全方位對照。
- **5.5 消融實驗（Ablation Study）**
  - 各模組增益分析（Dynamic Smash Gate, Sub-pixel, PMR, Strict Verify, 2-Hit Confirmation）。
  - 延遲、掉幀率與精準度三軸折衝分析（Pareto Frontier）。

---

### 第六章 結論與未來展望 (Chapter 6: Conclusion & Future Work)
- **6.1 研究結論與學術產出總結**
  - 實現 10W 邊緣嵌入式硬體上首個達成 30 FPS 零掉幀、Precision $\ge 95\%$ 之階層式羽球追蹤系統。
- **6.2 未來研究方向**
  - 3D 雙目立體視覺與鷹眼落點毫米級重建。
  - 結合 IMU 感測器與球拍姿態之擊球意圖預測。
  - 邊緣硬體轉移至 Jetson Orin Nano 之極限效能探討。

---

### 參考文獻 (References)
