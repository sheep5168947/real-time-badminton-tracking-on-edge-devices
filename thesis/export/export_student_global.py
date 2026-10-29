
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import os

try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError:
    print("[ERROR] Cannot find student_model.py")

def export_onnx():
    print("[INFO] Starting ONNX Export for 512x288...")
    
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 
    
    # 💡 關鍵修復：拔除 weights/ 資料夾前綴，直接存在當前目錄
    onnx_save_path = "miniroi_512x288.onnx"
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Cannot find weights at {weights_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniROIUNet().to(device)
    
    # 加上 weights_only=False 確保安全載入
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()

    # 建立 512x288 的 Dummy Input (注意：長寬順序是 H, W = 288, 512)
    dummy_input = torch.randn(1, 9, 288, 512).to(device)

    print("[INFO] Exporting Student Model (512x288) to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=12, # 保持與 128 模型一致
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"[INFO] Successfully exported ONNX to {onnx_save_path}")

if __name__ == '__main__':
    export_onnx()