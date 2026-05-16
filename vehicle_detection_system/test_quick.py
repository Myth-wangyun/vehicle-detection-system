#!/usr/bin/env python3
"""
快速测试脚本
验证所有模块功能，不依赖外部模型下载
"""
import os
import sys
import cv2
import numpy as np

# 设置环境变量（无 GUI 模式）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

print("=" * 60)
print("车辆检测系统 - 快速测试")
print("=" * 60)

# 1. 测试环境
print("\n[1/6] 测试环境...")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {'可用' if torch.cuda.is_available() else '不可用 (将使用 CPU)'}")

# 2. 测试 OpenCV
print("\n[2/6] 测试 OpenCV...")
import cv2
print(f"  OpenCV: {cv2.__version__}")

# 3. 测试项目模块
print("\n[3/6] 测试项目模块...")

from models.cbam import CBAM, DepthwiseSeparableConv
from models.diou_loss import DIoULoss
from tracker.deepsort_tracker import SimpleTracker
from tracker.speed_estimator import SpeedEstimator
from counter.line_counter import LineCounter

print("  CBAM 模块: OK")
print("  DIoU Loss: OK")
print("  SimpleTracker: OK")
print("  SpeedEstimator: OK")
print("  LineCounter: OK")

# 4. 测试模块功能
print("\n[4/6] 测试模块功能...")

# CBAM 测试
x = torch.randn(1, 256, 32, 32)
cbam = CBAM(256)
y = cbam(x)
print(f"  CBAM: {x.shape} -> {y.shape}")

# DIoU 测试
diou_loss = DIoULoss(loss_type='diou')
boxes1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
boxes2 = torch.tensor([[110, 110, 210, 210]], dtype=torch.float32)
loss = diou_loss(boxes1, boxes2)
print(f"  DIoU Loss: {loss.item():.4f}")

# 跟踪器测试
tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
detections = [[100, 100, 200, 200, 0.9, 2]]  # COCO class 2 = car
tracks = tracker.update(detections)
print(f"  跟踪器: {len(tracks)} 个跟踪目标")

# 速度估算测试
speed_est = SpeedEstimator(pixels_per_meter=30, fps=30)
speed_est.update(track_id=1, bbox=[100, 100, 200, 200], frame_idx=1)
speed = speed_est.update(track_id=1, bbox=[110, 100, 210, 200], frame_idx=2)
print(f"  速度估算: {speed:.2f} km/h")

# 跨线计数测试
line_cnt = LineCounter(line_start=(0, 240), line_end=(640, 240))
line_cnt.check_cross(track_id=1, bbox=[100, 200, 200, 250])
result = line_cnt.check_cross(track_id=1, bbox=[100, 250, 200, 300])
print(f"  跨线计数: {'成功' if result['crossed'] else '未触发'}")

# 5. 测试演示视频生成
print("\n[5/6] 测试演示视频生成...")

# 检查演示视频是否存在
demo_video = 'demo_videos/demo_traffic.mp4'
if os.path.exists(demo_video):
    print(f"  演示视频已存在: {demo_video}")
    # 读取一帧验证
    cap = cv2.VideoCapture(demo_video)
    ret, frame = cap.read()
    if ret:
        print(f"  视频读取成功: {frame.shape}")
    cap.release()
else:
    print("  生成演示视频...")
    # 使用现有脚本生成
    from create_demo_video import create_demo_video
    create_demo_video()
    print("  演示视频已生成!")

# 6. 测试 YOLO 模型
print("\n[6/6] 测试 YOLO 模型...")
try:
    from ultralytics import YOLO
    
    # 检查本地模型
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print("  正在下载 YOLOv8 预训练模型...")
        print("  (这是首次运行，Ultralytics 会自动下载约 6MB 的模型文件)")
    
    # 尝试加载（ultralytics 会自动处理下载）
    model = YOLO(model_path)
    print(f"  YOLO 模型加载成功!")
    
    # 创建测试图像
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 200), (400, 400), (0, 255, 0), -1)
    
    # 运行检测
    results = model(test_img, verbose=False)
    print(f"  YOLO 检测测试成功!")
    
except Exception as e:
    print(f"  YOLO 模型加载跳过: {e}")
    print("  (这不影响其他模块的正常使用)")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

print("\n下一步:")
print("  1. 运行演示: python demo.py")
print("  2. 启动 GUI: python main.py")
print("  3. 训练模型: python train.py")
print("  4. 打包发布: pyinstaller vehicle_detection.spec")
print("=" * 60)
