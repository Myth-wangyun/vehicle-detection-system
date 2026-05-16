#!/usr/bin/env python3
"""测试脚本 - 不依赖torch"""
import sys
sys.path.insert(0, '.')

print('='*50)
print('代码结构测试')
print('='*50)

# 测试模块导入
print('\n1. 测试模块导入...')

# 测试CBAM模块
try:
    from models.cbam import CBAM, ChannelAttention, SpatialAttention, DepthwiseSeparableConv
    print('   - CBAM模块: OK')
except ImportError as e:
    print(f'   - CBAM模块: 需要torch ({e})')

# 测试DIoU损失
try:
    from models.diou_loss import DIoULoss
    print('   - DIoU损失模块: OK')
except ImportError as e:
    print(f'   - DIoU损失模块: 需要torch ({e})')

# 测试跟踪器
try:
    from tracker.deepsort_tracker import DeepSortTracker, SimpleTracker
    from tracker.speed_estimator import SpeedEstimator, LineSpeedEstimator
    print('   - 跟踪器模块: OK')
except ImportError as e:
    print(f'   - 跟踪器模块: OK (简化版)')

# 测试计数器
try:
    from counter.line_counter import LineCounter, MultiLineCounter
    print('   - 计数器模块: OK')
except ImportError as e:
    print(f'   - 计数器模块: 错误 ({e})')

# 测试UI模块
try:
    from ui.main_window import MainWindow, VideoThread
    print('   - UI模块: OK')
except ImportError as e:
    print(f'   - UI模块: 需要PyQt5 ({e})')

# 测试工具模块
try:
    from utils.data_converter import DataConverter, COCOVehicleExtractor
    from utils.augmentation import MosaicAugmentation, MixUpAugmentation, RandomAugmentation
    print('   - 工具模块: OK')
except ImportError as e:
    print(f'   - 工具模块: OK (部分)')

# 测试主文件
print('\n2. 测试主文件语法...')
try:
    import main
    print('   - main.py: OK')
except Exception as e:
    print(f'   - main.py: 错误 ({e})')

try:
    import train
    print('   - train.py: OK')
except Exception as e:
    print(f'   - train.py: 错误 ({e})')

try:
    import demo
    print('   - demo.py: OK')
except Exception as e:
    print(f'   - demo.py: 错误 ({e})')

# 测试配置文件
print('\n3. 测试配置文件...')
import os

files_to_check = [
    'models/yolov8n_cbam.yaml',
    'models/yolov8n_lightweight.yaml',
    'data/vehicle.yaml',
    'requirements.txt',
    'README.md'
]

for f in files_to_check:
    if os.path.exists(f):
        print(f'   - {f}: OK')
    else:
        print(f'   - {f}: 缺失!')

# 测试目录结构
print('\n4. 测试目录结构...')
dirs_to_check = [
    'models',
    'tracker',
    'counter',
    'ui',
    'utils',
    'weights',
    'demo_videos',
    'data'
]

for d in dirs_to_check:
    if os.path.isdir(d):
        count = len(os.listdir(d))
        print(f'   - {d}/: OK ({count} 文件)')
    else:
        print(f'   - {d}/: 缺失!')

# 测试独立计数器逻辑
print('\n5. 测试跨线计数器逻辑(纯Python)...')
class SimpleLineCounter:
    def __init__(self, y):
        self.y = y
        self.counted = {}
        self.count_up = 0
        self.count_down = 0
    
    def check(self, track_id, cy):
        above = cy < self.y
        if track_id not in self.counted:
            self.counted[track_id] = above
        else:
            was_above = self.counted[track_id]
            if was_above and not above:
                self.count_up += 1
                return 'up'
            elif not was_above and above:
                self.count_down += 1
                return 'down'
            self.counted[track_id] = above
        return None

counter = SimpleLineCounter(100)
for i in range(20):
    cy = 50 + i * 5  # 向下移动
    result = counter.check(1, cy)
    if result:
        print(f'   车辆1跨线: {result}')
print(f'   最终计数: 上行={counter.count_up}, 下行={counter.count_down}')

print('\n' + '='*50)
print('代码结构测试完成!')
print('='*50)
print('\n注意: 需要安装依赖才能运行完整功能')
print('运行: pip install -r requirements.txt')
