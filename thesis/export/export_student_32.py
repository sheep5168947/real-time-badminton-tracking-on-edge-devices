
# Support direct script execution; relative data paths still use the project root.
if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
import os
import torch
from thesis.models.miniroi_unet import MiniROIUNet

def main():
    device = torch.device("cpu")
    model = MiniROIUNet().to(device)
    
    weight_path = "checkpoints_v4_student_32x32/epoch_30.pth"
    if not os.path.exists(weight_path):
        print(f"Error: {weight_path} does not exist.")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 9, 32, 32, device=device)
    os.makedirs("weights", exist_ok=True)
    onnx_path = "weights/miniroi_32x32.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"✅ 成功匯出 32x32 ONNX 模型至: {onnx_path}")

if __name__ == "__main__":
    main()
    
