#!/usr/bin/env python3
"""
生成合成交通场景数据集用于训练
生成图像 + YOLO 格式标注
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path

# 车辆类别和颜色
VEHICLE_CLASSES = {
    0: 'car',      # 汽车
    1: 'truck',    # 卡车
    2: 'bus',      # 公交车
    3: 'motorcycle' # 摩托车
}

VEHICLE_COLORS = {
    0: [(200, 50, 50), (50, 100, 200), (50, 180, 100), (200, 200, 50)],  # car
    1: [(150, 100, 50), (100, 50, 150), (50, 150, 150)],  # truck
    2: [(255, 200, 50), (200, 100, 50)],  # bus
    3: [(200, 200, 0), (150, 50, 200)]   # motorcycle
}

def draw_car(img, x, y, w, h, color):
    """绘制汽车"""
    # 车身
    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    # 车窗
    cv2.rectangle(img, (x+w//4, y+5), (x+3*w//4, y+h//2), (100, 150, 200), -1)
    # 车轮
    cv2.circle(img, (x+w//4, y+h), 10, (30, 30, 30), -1)
    cv2.circle(img, (x+3*w//4, y+h), 10, (30, 30, 30), -1)
    return img

def draw_truck(img, x, y, w, h, color):
    """绘制卡车"""
    # 驾驶室
    cv2.rectangle(img, (x, y), (x+w//3, y+h), color, -1)
    # 货箱
    cv2.rectangle(img, (x+w//3, y+5), (x+w, y+h-5), (100, 100, 100), -1)
    # 车轮
    for wx in [x+w//5, x+w//2, x+4*w//5]:
        cv2.circle(img, (wx, y+h), 12, (30, 30, 30), -1)
    return img

def draw_bus(img, x, y, w, h, color):
    """绘制公交车"""
    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    # 车窗
    for wx in range(x+10, x+w-15, 25):
        cv2.rectangle(img, (wx, y+8), (wx+18, y+h//2), (100, 150, 200), -1)
    # 车轮
    for wx in [x+20, x+w//3, x+2*w//3, x+w-20]:
        cv2.circle(img, (wx, y+h), 12, (30, 30, 30), -1)
    return img

def draw_motorcycle(img, x, y, w, h, color):
    """绘制摩托车"""
    # 车身
    cv2.ellipse(img, (x+w//2, y+h//2), (w//2, h//3), 0, 0, 360, color, -1)
    # 车轮
    cv2.circle(img, (x+10, y+h//2), 8, (30, 30, 30), -1)
    cv2.circle(img, (x+w-10, y+h//2), 8, (30, 30, 30), -1)
    return img

def draw_vehicle(img, x, y, w, h, class_id, color):
    """根据类别绘制车辆"""
    if class_id == 0:
        return draw_car(img, x, y, w, h, color)
    elif class_id == 1:
        return draw_truck(img, x, y, w, h, color)
    elif class_id == 2:
        return draw_bus(img, x, y, w, h, color)
    else:
        return draw_motorcycle(img, x, y, w, h, color)

def generate_scene(width=640, height=480, num_vehicles=3):
    """生成一个交通场景"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 50
    
    # 道路
    road_y1, road_y2 = height//3, 2*height//3
    cv2.rectangle(img, (0, road_y1), (width, road_y2), (80, 80, 80), -1)
    
    # 车道线
    for x in range(0, width, 50):
        cv2.rectangle(img, (x, (road_y1+road_y2)//2 - 2), (x+25, (road_y1+road_y2)//2 + 2), (200, 200, 200), -1)
    
    annotations = []
    
    # 随机放置车辆
    for _ in range(num_vehicles):
        class_id = random.choice([0, 0, 0, 1, 2, 3])  # car 更常见
        color = random.choice(VEHICLE_COLORS[class_id])
        
        # 随机尺寸
        if class_id == 0:  # car
            w, h = random.randint(60, 100), random.randint(35, 50)
        elif class_id == 1:  # truck
            w, h = random.randint(80, 120), random.randint(45, 60)
        elif class_id == 2:  # bus
            w, h = random.randint(100, 150), random.randint(50, 70)
        else:  # motorcycle
            w, h = random.randint(30, 50), random.randint(25, 35)
        
        # 随机位置（确保在道路范围内）
        x = random.randint(10, width - w - 10)
        y = random.randint(road_y1 + 10, road_y2 - h - 10)
        
        # 绘制
        img = draw_vehicle(img, x, y, w, h, class_id, color)
        
        # YOLO 格式标注: class_id x_center y_center width height (归一化)
        x_center = (x + w/2) / width
        y_center = (y + h/2) / height
        norm_w = w / width
        norm_h = h / height
        
        annotations.append((class_id, x_center, y_center, norm_w, norm_h))
    
    return img, annotations

def generate_dataset(output_dir, num_images=500, train_split=0.8):
    """生成完整数据集"""
    output_dir = Path(output_dir)
    
    # 创建目录
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    train_count = int(num_images * train_split)
    
    print(f"生成 {num_images} 张图像...")
    
    for i in range(num_images):
        # 生成图像
        num_vehicles = random.randint(2, 5)
        img, annotations = generate_scene(num_vehicles=num_vehicles)
        
        # 决定是训练集还是验证集
        is_train = i < train_count
        split = 'train' if is_train else 'val'
        
        # 保存图像
        img_path = output_dir / 'images' / split / f'{i:05d}.jpg'
        cv2.imwrite(str(img_path), img)
        
        # 保存标注
        label_path = output_dir / 'labels' / split / f'{i:05d}.txt'
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i+1}/{num_images} 张")
    
    print(f"数据集生成完成!")
    print(f"  训练集: {train_count} 张")
    print(f"  验证集: {num_images - train_count} 张")
    print(f"  保存位置: {output_dir}")
    
    return output_dir

if __name__ == '__main__':
    import sys
    
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'datasets/synthetic'
    
    generate_dataset(output_dir, num_images)
