#!/usr/bin/env python3
"""测试脚本"""
import sys
sys.path.insert(0, '.')

# 测试CBAM模块
print('测试CBAM模块...')
from models.cbam import CBAM, ChannelAttention, SpatialAttention, DepthwiseSeparableConv
import torch
x = torch.randn(1, 256, 32, 32)
cbam = CBAM(256)
y = cbam(x)
print(f'CBAM输入形状: {x.shape}, 输出形状: {y.shape}')
print('CBAM模块测试通过!')

# 测试DIoU损失
print('\n测试DIoU损失...')
from models.diou_loss import DIoULoss
pred_boxes = torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]])
target_boxes = torch.tensor([[105, 105, 205, 205], [60, 60, 160, 160]])
loss_fn = DIoULoss(loss_type='diou')
loss = loss_fn(pred_boxes, target_boxes)
print(f'DIoU Loss: {loss:.4f}')
print('DIoU损失测试通过!')

# 测试速度估算器
print('\n测试速度估算器...')
from tracker.speed_estimator import SpeedEstimator
estimator = SpeedEstimator(pixels_per_meter=30, fps=30)
for i in range(20):
    bbox = [100 + i*5, 200, 150 + i*5, 250]
    speed = estimator.update(1, bbox, i)
print(f'平均速度: {estimator.get_average_speed(1):.2f} km/h')
print('速度估算器测试通过!')

# 测试跨线计数器
print('\n测试跨线计数器...')
from counter.line_counter import LineCounter
counter = LineCounter(line_start=(0, 360), line_end=(1280, 360))
for i in range(50):
    bbox = [100, 200 + i*3, 200, 280 + i*3]
    result = counter.check_cross(1, bbox, speed=60, timestamp=i*0.03)
    if result['crossed']:
        print(f'跨线! 方向: {result["direction"]}')
print(f'最终计数: {counter.get_count()}')
print('跨线计数器测试通过!')

print('\n' + '='*50)
print('所有模块测试通过!')
print('='*50)
