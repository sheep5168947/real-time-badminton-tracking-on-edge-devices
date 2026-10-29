
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
import torch
from thesis.models.miniroi_unet import MiniROIUNet

def export_to_onnx():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 載入模型架構與 64x64 權重
    model = MiniROIUNet().to(device)
    weights_path = "checkpoints_v4_student_64x64/epoch_30.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    
    # 2. 建立 64x64 的 Dummy Input
    # 由於這次是做嚴格的 64x64 極限測試，我們不需要設置動態尺寸 (dynamic_axes)
    dummy_input = torch.randn(1, 9, 64, 64, device=device)
    onnx_file_path = "student_model_64x64.onnx"
    
    # 3. 匯出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print("ONNX 匯出成功！檔案已存為: {}".format(onnx_file_path))

if __name__ == '__main__':
    export_to_onnx()