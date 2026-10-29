import os
import cv2
import time

def test_gstreamer():
    video_path = "weights/test_video.mp4"
    abs_path = os.path.abspath(video_path)
    
    if not os.path.exists(abs_path):
        print(f"[ERROR] File not found: {abs_path}")
        return

    print(f"[INFO] Testing GStreamer with: {abs_path}")
    
    # 使用 uridecodebin 讓 GStreamer 自動判斷格式並調用硬體解碼
    gstreamer_pipeline = (
        f"uridecodebin uri=file://{abs_path} ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )
    
    print("[INFO] Pipeline string:")
    print(gstreamer_pipeline)
    
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[ERROR] GStreamer failed to open the pipeline.")
        # 檢查 OpenCV 是否編譯了 GStreamer 支援
        print("\n[DEBUG] Checking OpenCV Build Information for GStreamer...")
        build_info = cv2.getBuildInformation()
        for line in build_info.split('\n'):
            if "GStreamer" in line:
                print(line)
        return

    print("[INFO] GStreamer Pipeline Opened Successfully!")
    
    frame_count = 0
    start_time = time.perf_counter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 只讀取前 100 幀來測速
        if frame_count >= 100:
            break

    elapsed_time = time.perf_counter() - start_time
    fps = frame_count / elapsed_time
    
    cap.release()
    print(f"[INFO] Decoded {frame_count} frames in {elapsed_time:.2f} seconds.")
    print(f"[INFO] Hardware Decoding Speed: {fps:.2f} FPS")

if __name__ == "__main__":
    test_gstreamer()