# -*- coding: utf-8 -*-
"""
Export MiniROIUNet Student Model for Fast Global Rescue (256x144) to ONNX.
Input Shape: (1, 9, 144, 256) [B, C, H, W]
Output Shape: (1, 1, 144, 256)
"""

if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import torch
import os

try:
    from thesis.models.miniroi_unet import MiniROIUNet
except ImportError:
    print("[ERROR] Cannot find MiniROIUNet")

def export_onnx(output_dir="weights"):
    print("[INFO] Starting ONNX Export for Fast Global Rescue (256x144)...")
    
    weights_path = r"checkpoints_v4_student\epoch_30.pth" 
    if not os.path.isabs(weights_path):
        root_dir = str(_ThesisPath(__file__).resolve().parents[2])
        weights_path = os.path.join(root_dir, weights_path)

    os.makedirs(output_dir, exist_ok=True)
    onnx_save_path = os.path.join(output_dir, "miniroi_256x144.onnx")
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Cannot find weights at {weights_path}")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniROIUNet().to(device)
    
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()

    # Dummy Input for 256x144 (H=144, W=256)
    dummy_input = torch.randn(1, 9, 144, 256).to(device)

    print(f"[INFO] Exporting Student Model (256x144) to {onnx_save_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"[INFO] Successfully exported ONNX to {onnx_save_path}")
    return onnx_save_path

if __name__ == '__main__':
    export_onnx()
