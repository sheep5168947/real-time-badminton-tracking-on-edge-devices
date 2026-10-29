import cv2
import numpy as np
import time
import onnxruntime as ort
from collections import deque
import os

def test_onnx_fps():
    # 檔案路徑設定 (請確認檔名與路徑正確)
    onnx_path = "weights/tracknet_v4_512x288.onnx"
    video_path = "weights/test_video.mp4"

    if not os.path.exists(onnx_path):
        print(f"找不到模型檔案: {onnx_path}")
        return
    if not os.path.exists(video_path):
        print(f"找不到影片檔案: {video_path}")
        return

    print("🚀 正在啟動 ONNX Runtime 推論引擎...")
    # 嘗試呼叫 TensorRT 或 CUDA 來硬體加速，如果都沒有就退回 CPU
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"✅ 模型載入成功！目前使用的加速器: {session.get_providers()[0]}")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    input_name = session.get_inputs()[0].name
    
    print(f"🎥 開始讀取影片: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # 準備 3 幀的滑動窗口
    frame_queue = deque(maxlen=3)
    
    # 先讀取前 2 幀來填滿窗口
    for _ in range(2):
        ret, frame = cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (512, 288))
        # BGR 轉 RGB 並換軸為 (C, H, W)，正規化到 0~1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_queue.append(frame_tensor)

    frame_count = 0
    print("🔥 開始全速推論測速 (這可能會讓 Nano 發出哀嚎)...")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.resize(frame, (512, 288))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_queue.append(frame_tensor)

        # 組合 3 幀變成 (9, 288, 512)，再增加 Batch 維度變成 (1, 9, 288, 512)
        input_data = np.concatenate(list(frame_queue), axis=0)
        input_data = np.expand_dims(input_data, axis=0)

        # 執行推論！
        _ = session.run(None, {input_name: input_data})
        
        # 每處理 10 幀印出一次當前進度
        if frame_count % 10 == 0:
            print(f"已處理 {frame_count} 幀...")

    end_time = time.time()
    cap.release()

    # 計算殘酷的成績單
    total_time = end_time - start_time
    fps = frame_count / total_time
    
    print("\n" + "="*40)
    print("📊 [Baseline 災難數據報表]")
    print("="*40)
    print(f"模型架構: TrackNetV4 (全圖 512x288)")
    print(f"處理總幀數: {frame_count} 幀")
    print(f"總耗時: {total_time:.2f} 秒")
    print(f"🔥 平均 FPS: {fps:.2f} fps")
    print(f"⏱️ 單幀推論耗時: {(1/fps)*1000:.2f} ms")
    print("="*40)
    print("把這個數據記下來，這就是你的論文 Baseline！")

if __name__ == "__main__":
    test_onnx_fps()