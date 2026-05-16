"""
模型注册脚本
注册自定义模块到 YOLO 模型中
"""
import torch.nn as nn
from ultralytics.nn.modules import *
from ultralytics.nn import tasks

# 导入自定义模块
from models.cbam import CBAM, CBAMBlock, DepthwiseSeparableConv


class DSConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dsconv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.dsconv(x)


class CBAMModule(nn.Module):
    """CBAM 模块封装"""
    def __init__(self, c1, c2=None):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        return self.cbam(x)


def register_modules():
    """注册自定义模块到全局模块字典"""
    # 注册到 nn 模块
    nn.CBAM = CBAM
    nn.CBAMBlock = CBAMBlock
    nn.DepthwiseSeparableConv = DepthwiseSeparableConv
    nn.DSConv = DSConv
    
    # 注册到 tasks 模块
    if hasattr(tasks, 'CBAM'):
        pass
    else:
        setattr(tasks, 'CBAM', CBAM)
        setattr(tasks, 'CBAMBlock', CBAMBlock)
        setattr(tasks, 'DepthwiseSeparableConv', DepthwiseSeparableConv)
        setattr(tasks, 'DSConv', DSConv)
    
    print("自定义模块注册完成: CBAM, CBAMBlock, DepthwiseSeparableConv, DSConv")


if __name__ == "__main__":
    # 测试模块注册
    register_modules()
    
    # 测试 CBAM
    import torch
    x = torch.randn(1, 256, 32, 32)
    cbam = CBAM(256)
    y = cbam(x)
    print(f"CBAM: 输入 {x.shape} -> 输出 {y.shape}")
    
    # 测试 DSConv
    dsconv = DSConv(64, 128)
    y = dsconv(x[:, :64, :, :])
    print(f"DSConv: 输入 {x[:, :64, :, :].shape} -> 输出 {y.shape}")
