import cv2
import numpy as np
from PIL import Image, ImageSequence
import os

def create_comparison():
    gif_native = Image.open(r"E:\Thesis\tracknet-series-pytorch\thesis_manuscript\assets\slide9_img_4.gif")
    native_frames = [np.array(f.copy().convert("RGB")) for f in ImageSequence.Iterator(gif_native)]
    
    cap = cv2.VideoCapture(r"E:\Thesis\tracknet-series-pytorch\report_materials\test_video_nano_realtime_proposed.mp4")
    prop_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prop_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    out_frames = []
    target_h = 240
    target_w = 426
    
    total_pairs = min(len(native_frames), len(prop_frames) // 2)
    print(f"Creating side-by-side comparison with {total_pairs} frames...")
    
    for i in range(total_pairs):
        f_nat = cv2.resize(native_frames[i], (target_w, target_h))
        prop_idx = min(i * 2, len(prop_frames) - 1)
        f_prop = cv2.resize(prop_frames[prop_idx], (target_w, target_h))
        
        # Add labels
        cv2.rectangle(f_nat, (0, 0), (target_w, 28), (15, 23, 42), -1)
        cv2.putText(f_nat, "Native TrackNetV4 (0.9 FPS | Huge Lag)", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA)
                    
        cv2.rectangle(f_prop, (0, 0), (target_w, 28), (15, 23, 42), -1)
        cv2.putText(f_prop, "Proposed Tracker (30.0 FPS | Zero Drop)", (10, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (52, 211, 153), 1, cv2.LINE_AA)
                    
        # Concatenate horizontally
        combined = np.hstack([f_nat, f_prop])
        im_comb = Image.fromarray(combined).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        out_frames.append(im_comb)
        
    out_path = r"E:\Thesis\tracknet-series-pytorch\report_materials\test_video_native_vs_proposed_comparison.gif"
    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=66, # ~15 FPS sync
        loop=0,
        optimize=True
    )
    
    sz = os.path.getsize(out_path)
    print(f"Comparison GIF saved: {out_path} ({sz / (1024*1024):.2f} MB, {len(out_frames)} frames)")
    
    # Also copy to assets and brain
    shutil_copy(out_path, r"E:\Thesis\tracknet-series-pytorch\thesis_manuscript\assets\test_video_native_vs_proposed_comparison.gif")
    shutil_copy(out_path, r"C:\Users\User\.gemini\antigravity\brain\dac42c7a-e416-43b0-ae7b-dc7800d2667b\test_video_native_vs_proposed_comparison.gif")

def shutil_copy(src, dst):
    import shutil
    shutil.copy(src, dst)

if __name__ == "__main__":
    create_comparison()
