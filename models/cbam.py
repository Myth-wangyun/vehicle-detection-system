"""
CBAM (Convolutional Block Attention Module) 模块
用于YOLOv8的特征增强
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class CBAM(nn.Module):
    """CBAM注意力模块 - 通道注意力和空间注意力的串行组合"""
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CBAMBlock(nn.Module):
    """可嵌入到YOLOv8 C2f模块后的CBAM块"""
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.cbam = CBAM(in_channels, ratio, kernel_size)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        out = self.cbam(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out + x  # 残差连接


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 用于模型轻量化"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    # 测试CBAM模块
    print("测试CBAM模块...")
    x = torch.randn(1, 256, 32, 32)
    cbam = CBAM(256)
    y = cbam(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试CBAMBlock
    cbam_block = CBAMBlock(256)
    y = cbam_block(x)
    print(f"CBAMBlock输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试深度可分离卷积
    ds_conv = DepthwiseSeparableConv(256, 128)
    y = ds_conv(x)
    print(f"深度可分离卷积输入形状: {x.shape}, 输出形状: {y.shape}")
    
    print("CBAM模块测试通过!")
