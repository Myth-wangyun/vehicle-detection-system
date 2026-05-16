"""
车辆检测与速度检测系统
基于改进YOLOv8算法
"""
__version__ = "1.0.0"
__author__ = "Wang Yun"

from .models.cbam import CBAM, CBAMBlock, ChannelAttention, SpatialAttention
from .models.diou_loss import DIoULoss, YOLOv8Loss
from .tracker.speed_estimator import SpeedEstimator, LineSpeedEstimator
from .counter.line_counter import LineCounter, MultiLineCounter

__all__ = [
    'CBAM',
    'CBAMBlock', 
    'ChannelAttention',
    'SpatialAttention',
    'DIoULoss',
    'YOLOv8Loss',
    'SpeedEstimator',
    'LineSpeedEstimator',
    'LineCounter',
    'MultiLineCounter',
]
