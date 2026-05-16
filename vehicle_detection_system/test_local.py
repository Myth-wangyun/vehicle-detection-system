#!/usr/bin/env python3
"""
车辆检测系统 - 本地测试脚本
在本地电脑上运行，可以下载真实图片测试

使用方法:
    python test_local.py
    
    或者指定图片:
    python test_local.py --image your_car_photo.jpg
"""

import argparse
import os
import urllib.request
import cv2
import numpy as np

# 尝试导入 yolov8
try:
    from ultralytics import YOLO
except ImportError:
    print("需要安装 ultralytics: pip install ultralytics")
    exit(1)

# 测试图片 URLs (GitHub 公开图片)
TEST_IMAGES = [
    ("https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=640", "highway.jpg"),
    ("https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=640", "car1.jpg"),
    ("https://images.unsplash.com/photo-1502877338535-766e1452684a?w=640", "car2.jpg"),
    ("https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=640", "sports_car.jpg"),
]

def download_image(url, filename):
    """下载图片"""
    try:
        print(f"下载: {url[:60]}...")
        urllib.request.urlretrieve(url, filename)
        size = os.path.getsize(filename)
        print(f"  ✅ 成功: {filename} ({size/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return False

def test_yolo_detection(image_path):
    """测试 YOLO 检测"""
    print(f"\n加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')
    
    print(f"检测图片: {image_path}")
    results = model(image_path, conf=0.3, verbose=True)
    
    # 绘制结果
    img = cv2.imread(image_path)
    
    vehicle_classes = {
        'car': 0, 'truck': 7, 'bus': 5, 'motorcycle': 3,
        'bicycle': 1, 'person': 0, 'dog': 16
    }
    
    vehicles = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'other': 0}
    total = 0
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        cls_name = model.names[cls_id]
        
        # 只统计车辆相关
        if cls_name.lower() in ['car', 'truck', 'bus', 'motorcycle']:
            vehicles[cls_name.lower()] += 1
            total += 1
        elif cls_name.lower() in ['person', 'bicycle']:
            vehicles['other'] += 1
            total += 1
        
        # 画框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if cls_name.lower() in ['car', 'truck', 'bus', 'motorcycle'] else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 保存结果
    result_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(result_path, img)
    
    return vehicles, total, result_path

def main():
    parser = argparse.ArgumentParser(description='车辆检测系统本地测试')
    parser.add_argument('--image', '-i', type=str, help='指定测试图片路径')
    parser.add_argument('--download', '-d', action='store_true', help='下载测试图片')
    args = parser.parse_args()
    
    print("=" * 60)
    print("车辆检测系统 - 本地测试")
    print("=" * 60)
    
    if args.image and os.path.exists(args.image):
        # 使用指定图片
        vehicles, total, result_path = test_yolo_detection(args.image)
        print(f"\n检测结果: {total} 个车辆")
        for vtype, count in vehicles.items():
            if count > 0:
                print(f"  - {vtype}: {count}")
        print(f"\n结果图片: {result_path}")
        
    elif args.download:
        # 下载测试图片
        print("\n下载测试图片...")
        for url, filename in TEST_IMAGES:
            if download_image(url, filename):
                vehicles, total, result_path = test_yolo_detection(filename)
                print(f"\n检测结果: {total} 个车辆")
                for vtype, count in vehicles.items():
                    if count > 0:
                        print(f"  - {vtype}: {count}")
                print(f"结果图片: {result_path}")
                print()
                break
        
    else:
        # 交互式
        print("\n请选择:")
        print("1. 使用指定图片测试")
        print("2. 下载测试图片")
        print("3. 创建演示图片")
        
        choice = input("\n输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            img_path = input("图片路径: ").strip()
            if os.path.exists(img_path):
                vehicles, total, result_path = test_yolo_detection(img_path)
                print(f"\n检测结果: {total} 个车辆")
            else:
                print("文件不存在!")
        
        elif choice == '2':
            for url, filename in TEST_IMAGES:
                if download_image(url, filename):
                    vehicles, total, result_path = test_yolo_detection(filename)
                    print(f"\n检测结果: {total} 个车辆")
                    break
        
        elif choice == '3':
            # 创建演示
            print("\n创建演示图片...")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = (100, 120, 100)  # 天空
            
            # 道路
            img[250:, :] = (80, 80, 80)
            
            # 画更真实的车辆
            cv2.rectangle(img, (100, 280), (220, 350), (180, 50, 50), -1)
            cv2.rectangle(img, (120, 290), (200, 320), (60, 100, 140), -1)
            cv2.circle(img, (130, 350), 20, (30, 30, 30), -1)
            cv2.circle(img, (190, 350), 20, (30, 30, 30), -1)
            
            cv2.rectangle(img, (300, 280), (450, 350), (50, 100, 200), -1)
            cv2.rectangle(img, (320, 290), (380, 320), (60, 100, 140), -1)
            cv2.circle(img, (330, 350), 20, (30, 30, 30), -1)
            cv2.circle(img, (420, 350), 20, (30, 30, 30), -1)
            
            cv2.imwrite('demo_input.jpg', img)
            print("演示图片已保存: demo_input.jpg")
            
            vehicles, total, result_path = test_yolo_detection('demo_input.jpg')
            print(f"\n检测结果: {total} 个车辆")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
