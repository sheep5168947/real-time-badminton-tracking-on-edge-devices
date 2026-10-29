import torch
import time
from tracknet.models.tracknet_v4 import TrackNetV4

def benchmark_model():
    # 確認是否抓到你的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 目前使用的運算設備: {device}")

    # 1. 建立空殼模型 (無權重)
    print("正在建立 TrackNetV4 模型 (隨機權重)...")
    model = TrackNetV4()
    model = model.to(device)
    model.eval() # 記得設定為推論模式

    # 2. 準備假影像資料 (Dummy Input)
    # TrackNet 通常一次吃 3 張連續影像 (RGB 共 9 通道)，常見解析度為 288x512
    print("準備模擬輸入影像 (Batch=1, Channels=9, Height=288, Width=512)...")
    dummy_input = torch.randn(1, 9, 288, 512).to(device)

    # 3. GPU 暖機 (Warm-up)
    print("GPU 暖機中...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 4. 正式測速 (跑 100 次取平均)
    print("開始測量推論速度 (跑 100 次)...")
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    end_time = time.time()

    # 5. 計算並輸出 FPS
    total_time = end_time - start_time
    avg_time_per_frame = total_time / num_iterations
    fps = 1.0 / avg_time_per_frame

    print("=" * 40)
    print("✅ 測試完成！")
    print(f"總共耗時: {total_time:.2f} 秒")
    print(f"單次推論耗時: {avg_time_per_frame * 1000:.2f} ms")
    print(f"👉 PC 端預估 FPS: {fps:.2f} 幀/秒")
    print("=" * 40)

if __name__ == '__main__':
    benchmark_model()