
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import os
import sys

try:
    from tracknet.models.tracknet_v4 import TrackNetV4
except ImportError:
    print("Error: Cannot find tracknet module. Ensure you run this from the root directory.")
    sys.exit(1)

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 建立 V4 模型空殼
    model = TrackNetV4().to(device)
    
    # 2. 載入你剛訓練好的 V4 王牌導師權重
    # 注意：請確認這裡的路徑跟你昨天產出的 best.pt 路徑一致！
    weights_path = r"outputs\train\tracknet_v4_20260715_075947\checkpoints\best.pt"
    if not os.path.exists(weights_path):
        print(f"找不到權重檔案：{weights_path}")
        return

    print("Loading weights...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    # 切換為推理模式 (重要！)
    model.eval()

    # 3. 建立一個虛擬的輸入張量 (Dummy Input) 來讓 ONNX 追蹤計算圖
    # V4 的輸入是連續 3 幀，每幀 3 個 Channel (RGB)，所以深度是 9
    # 解析度是我們設定的 512 寬, 288 高
    dummy_input = torch.randn(1, 9, 288, 512).to(device)
    onnx_filename = "tracknet_v4_512x288.onnx"

    # 4. 正式導出 ONNX
    print("Exporting model to ONNX format... (This might take a minute)")
    torch.onnx.export(
        model,                     # 欲導出的模型
        dummy_input,               # 虛擬輸入
        onnx_filename,             # 輸出的檔名
        export_params=True,        # 將權重一併導出
        opset_version=12,          # ONNX 的版本 (12 對 Jetson 支援度很好)
        do_constant_folding=True,  # 優化計算圖
        input_names=['input'],     # 設定輸入層名稱
        output_names=['output'],   # 設定輸出層名稱
    )
    
    print(f"🎉 轉換成功！檔案已儲存為：{onnx_filename}")
    print("接下來請將此 .onnx 檔案傳輸到 Jetson Nano 上。")

if __name__ == "__main__":
    export_to_onnx()