import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分離卷積：大幅降低參數與運算量，Edge AI 必備神兵利器
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MiniROIUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 輸入：3 張 RGB 圖片 (Channels = 9)
        
        # --- Encoder (下採樣萃取特徵) ---
        self.enc1 = DepthwiseSeparableConv(9, 32)      # 128x128
        self.pool1 = nn.MaxPool2d(2)                   # -> 64x64
        
        self.enc2 = DepthwiseSeparableConv(32, 64)     # 64x64
        self.pool2 = nn.MaxPool2d(2)                   # -> 32x32
        
        self.enc3 = DepthwiseSeparableConv(64, 128)    # 32x32
        self.pool3 = nn.MaxPool2d(2)                   # -> 16x16

        # --- Bottleneck ---
        self.bottleneck = DepthwiseSeparableConv(128, 128) # 16x16

        # --- Decoder (上採樣還原熱力圖) ---
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2) # -> 32x32
        # 修改：128 (來自 up1) + 128 (來自 e3) = 256
        self.dec1 = DepthwiseSeparableConv(256, 64)  
        
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2) # -> 64x64
        # 修改：64 (來自 d1) + 64 (來自 e2) = 128
        self.dec2 = DepthwiseSeparableConv(128, 32)
        
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2) # -> 128x128
        # 修改：32 (來自 d2) + 32 (來自 e1) = 64
        self.dec3 = DepthwiseSeparableConv(64, 16)
        
        # --- Output Layer ---
        # 輸出 1 張熱力圖 (Channel = 1)，用 Sigmoid 壓到 0~1 之間
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder with Skip Connections (Concat 拼接特徵)
        d1 = self.up1(b)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1) # 補上這行！把 e1 的特徵也接過來
        d3 = self.dec3(d3)
        
        out = self.final_conv(d3)
        return self.sigmoid(out)

# 測試模型輸入輸出大小
if __name__ == '__main__':
    model = MiniROIUNet()
    # 模擬 1 個 Batch, 9 個 Channel, 128x128 的輸入
    dummy_input = torch.randn(1, 9, 128, 128) 
    output = model(dummy_input)
    print(f"輸入大小: {dummy_input.shape}")
    print(f"輸出大小: {output.shape}")