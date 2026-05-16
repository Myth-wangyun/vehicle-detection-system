#!/bin/bash
# 车辆检测系统 - 一键测试脚本
# 在有网络的本地电脑上运行

echo "=========================================="
echo "车辆检测系统 - 本地测试"
echo "=========================================="

# 1. 检查 Python
echo ""
echo "[1/5] 检查 Python 环境..."
python3 --version || { echo "需要安装 Python 3"; exit 1; }

# 2. 安装依赖
echo ""
echo "[2/5] 安装依赖..."
pip install ultralytics opencv-python torch torchvision

# 3. 下载测试图片
echo ""
echo "[3/5] 下载测试图片..."
mkdir -p test_images
cd test_images

# 使用 curl 下载公开图片
echo "下载 highway 图片..."
curl -sL "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/movingcar.jpg" -o highway.jpg 2>/dev/null || \
curl -sL "https://people.cs.uchicago.edu/~hanbyuly/hci/sun_a准入/131.png" -o test1.jpg 2>/dev/null || \
echo "下载失败，请手动提供测试图片"

cd ..

# 4. 运行检测测试
echo ""
echo "[4/5] 运行检测测试..."
python3 << 'PYTHON'
import cv2
from ultralytics import YOLO

print("加载 YOLOv8n 模型...")
model = YOLO('yolov8n.pt')

# 测试目录中的图片
import os
test_dir = 'test_images'
if os.path.exists(test_dir):
    for img_file in os.listdir(test_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_dir, img_file)
            print(f"\n检测: {img_file}")
            results = model(img_path, conf=0.3)
            detections = len(results[0].boxes)
            print(f"  检测到 {detections} 个对象")
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)
                print(f"    - {cls_name}: {conf:.2f}")
            
            # 保存结果
            results[0].save(f'result_{img_file}')
            print(f"  结果已保存: result_{img_file}")

# 创建测试图片（如果没有下载成功）
print("\n创建测试图片...")
img = cv2.imread('test_images/highway.jpg') if os.path.exists('test_images/highway.jpg') else None

if img is None:
    import numpy as np
    # 创建简单测试图
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (0, 200), (640, 400), (100, 100, 100), -1)
    
    # 添加车辆
    for i, (x, color) in enumerate([(50, (50, 50, 200)), (200, (200, 50, 50)), (350, (50, 150, 50))]):
        y = 250 + (i % 2) * 50
        cv2.rectangle(img, (x, y), (x+100, y+60), color, -1)
        cv2.circle(img, (x+20, y+60), 15, (30, 30, 30), -1)
        cv2.circle(img, (x+80, y+60), 15, (30, 30, 30), -1)
    
    cv2.imwrite('test_images/test.jpg', img)
    print("测试图片已创建")
    
    results = model('test_images/test.jpg', conf=0.25)
    print(f"检测到 {len(results[0].boxes)} 个对象")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
PYTHON

# 5. 运行完整演示
echo ""
echo "[5/5] 运行完整演示..."
python3 demo.py --video demo_videos/real_detection_test.mp4 --headless || \
python3 demo.py --headless

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
