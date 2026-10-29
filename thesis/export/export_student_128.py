
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import os

# 根據你的實際檔案架構引入你的學生模型 (MiniROIUNet)
try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError:
    print("找不到 student_model，請確認你的子模型檔名！")
    
def export_student_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniROIUNet().to(device)
    
    # 這裡請改成你 checkpoints_v4_student 資料夾中最新/最大的那個權重檔
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 
    
    if not os.path.exists(weights_path):
        print(f"找不到權重檔案：{weights_path}")
        return

    print("Loading weights...")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    # 💎 關鍵差異：學生模型只看 128x128 的小區塊！
    dummy_input = torch.randn(1, 9, 128, 128).to(device)
    onnx_filename = "miniroi_128x128.onnx"

    print("Exporting Student Model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    
    print(f"🎉 轉換成功！檔案已儲存為：{onnx_filename}")

if __name__ == "__main__":
    export_student_to_onnx()