# -*- coding: utf-8 -*-
"""
Export Retrained Augmented MiniROI Models to ONNX:
1. miniroi_128x128_augmented.onnx (Local Fine Tracker)
2. miniroi_256x144_augmented.onnx (Global Fast Rescue)
"""

if __package__ in (None, ""):
    import sys as _thesis_sys
    from pathlib import Path as _ThesisPath
    _thesis_sys.path.insert(0, str(_ThesisPath(__file__).resolve().parents[2]))

import os
import torch
from thesis.models.miniroi_unet import MiniROIUNet

def export_augmented():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = r"checkpoints_v4_student_augmented\best_augmented.pth"
    output_dir = "weights"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"[ERROR] Cannot find weights at {weights_path}")
        return
        
    print(f"[EXPORT] Loading best augmented model from {weights_path}...")
    model = MiniROIUNet().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # 1. Export 128x128
    dummy_128 = torch.randn(1, 9, 128, 128).to(device)
    out_128 = os.path.join(output_dir, "miniroi_128x128_augmented.onnx")
    print(f"[EXPORT] Exporting 128x128 to {out_128}...")
    torch.onnx.export(
        model, dummy_128, out_128,
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )
    
    # 2. Export 256x144
    dummy_256 = torch.randn(1, 9, 144, 256).to(device)
    out_256 = os.path.join(output_dir, "miniroi_256x144_augmented.onnx")
    print(f"[EXPORT] Exporting 256x144 to {out_256}...")
    torch.onnx.export(
        model, dummy_256, out_256,
        export_params=True, opset_version=12, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )
    
    print("[SUCCESS] Both augmented models exported to ONNX successfully!")

if __name__ == '__main__':
    export_augmented()
