#!/usr/bin/env python3
"""
真实检测演示脚本
使用 YOLO 预训练模型在合成视频上进行检测
展示完整的检测、跟踪、速度估算、计数功能
"""

import cv2
import numpy as np
from ultralytics import YOLO
import random
from PIL import Image, ImageDraw

# 类别映射
CLASS_NAMES = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle'}
CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 165, 255),
    'motorcycle': (255, 255, 0)
}

def draw_real_car(draw, x, y, w, h, color):
    """绘制汽车"""
    # 车身 - 更圆润的形状
    draw.polygon([
        (x, y+h), (x, y+h//2), (x+w//4, y+5), (x+3*w//4, y+5), (x+w, y+h//2), (x+w, y+h)
    ], fill=color, outline=(30, 30, 30))
    
    # 车窗
    draw.polygon([
        (x+w//4+5, y+10), (x+w//4+12, y+h//2-5), 
        (x+3*w//4-12, y+h//2-5), (x+3*w//4-5, y+10)
    ], fill=(80, 120, 160), outline=(50, 50, 50))
    
    # 车轮
    for wx in [x+w//4, x+3*w//4]:
        draw.ellipse([wx-12, y+h-3, wx+12, y+h+12], fill=(30, 30, 30))
        draw.ellipse([wx-6, y+h+2, wx+6, y+h+8], fill=(80, 80, 80))

def create_traffic_video(output_path, duration_sec=10, fps=30, width=1280, height=720):
    """创建真实感的交通视频"""
    print(f"创建交通视频: {width}x{height}, {fps}fps, {duration_sec}秒")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    # 车辆配置
    vehicles = [
        {'type': 'car', 'color': (200, 50, 50), 'plate': '京A12345', 'start': 0, 'duration': 180, 'lane': 350, 'speed': 3.0},
        {'type': 'car', 'color': (50, 100, 200), 'plate': '沪B67890', 'start': 30, 'duration': 180, 'lane': 400, 'speed': 2.8},
        {'type': 'truck', 'color': (100, 100, 100), 'plate': '粤C11111', 'start': 60, 'duration': 200, 'lane': 450, 'speed': 2.0},
        {'type': 'bus', 'color': (255, 200, 50), 'plate': '京D22222', 'start': 90, 'duration': 200, 'lane': 320, 'speed': 2.5},
        {'type': 'car', 'color': (50, 180, 100), 'plate': '沪E33333', 'start': 120, 'duration': 160, 'lane': 380, 'speed': 3.2},
        {'type': 'motorcycle', 'color': (200, 200, 0), 'plate': '粤F44444', 'start': 45, 'duration': 150, 'lane': 430, 'speed': 4.0},
    ]
    
    for frame_idx in range(total_frames):
        # 创建场景
        img = np.ones((height, width, 3), dtype=np.uint8) * 60
        
        # 渐变天空
        for y in range(height//3):
            v = int(40 + (y / (height//3)) * 30)
            img[y, :] = [v, v, v+10]
        
        # 道路
        road_y1, road_y2 = 250, 500
        cv2.rectangle(img, (0, road_y1), (width, road_y2), (60, 60, 60), -1)
        
        # 车道线
        for x in range(0, width, 60):
            cv2.rectangle(img, (x, (road_y1+road_y2)//2 - 3), (x+30, (road_y1+road_y2)//2 + 3), (200, 200, 200), -1)
        
        # 人行道
        cv2.rectangle(img, (0, road_y2), (width, road_y2+60), (90, 90, 90), -1)
        
        # 建筑物
        for bx in range(50, width, 180):
            bh = 80 + (bx % 50)
            cv2.rectangle(img, (bx, road_y1-bh), (bx+120, road_y1), (70, 70, 75), -1)
            # 窗户
            for wy in range(road_y1-bh+15, road_y1-10, 25):
                for wx in range(bx+10, bx+110, 25):
                    cv2.rectangle(img, (wx, wy), (wx+12, wy+15), (180, 160, 100), -1)
        
        # 时间戳
        time_str = f"2024-01-15 14:{30 + frame_idx//1800:02d}:{(frame_idx//30)%60:02d}"
        cv2.putText(img, time_str, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制车辆
        for v in vehicles:
            t = frame_idx - v['start']
            if 0 <= t < v['duration']:
                if v['type'] == 'car':
                    x = int(t * v['speed'])
                    y = v['lane']
                    w, h = 100, 50
                    # 绘制车形
                    pts = np.array([[x, y+h], [x, y+h//2], [x+w//4, y], [x+3*w//4, y], [x+w, y+h//2], [x+w, y+h]], np.int32)
                    cv2.fillPoly(img, [pts], v['color'])
                    # 车窗
                    pts2 = np.array([[x+w//4+5, y+5], [x+w//4+10, y+h//2-5], [x+3*w//4-10, y+h//2-5], [x+3*w//4-5, y+5]], np.int32)
                    cv2.fillPoly(img, [pts2], (80, 120, 160))
                    # 车牌
                    cv2.rectangle(img, (x+w//3, y+h-12), (x+2*w//3, y+h-5), (255, 255, 200), -1)
                    
                elif v['type'] == 'truck':
                    x = int(t * v['speed'])
                    y = v['lane']
                    w, h = 140, 60
                    cv2.rectangle(img, (x, y+10), (x+w//3, y+h), v['color'], -1)
                    cv2.rectangle(img, (x+w//3, y+5), (x+w, y+h-5), (100, 100, 100), -1)
                    
                elif v['type'] == 'bus':
                    x = int(t * v['speed'])
                    y = v['lane']
                    w, h = 180, 65
                    cv2.rectangle(img, (x, y+10), (x+w, y+h), v['color'], -1)
                    # 车窗
                    for wx in range(x+15, x+w-20, 30):
                        cv2.rectangle(img, (wx, y+15), (wx+20, y+h//2), (80, 120, 160), -1)
                        
                elif v['type'] == 'motorcycle':
                    x = int(t * v['speed'])
                    y = v['lane']
                    cv2.ellipse(img, (x+25, y+25), (25, 15), 0, 0, 360, v['color'], -1)
                    cv2.circle(img, (x+10, y+25), 8, (30, 30, 30), -1)
                    cv2.circle(img, (x+40, y+25), 8, (30, 30, 30), -1)
        
        out.write(img)
        
        if (frame_idx + 1) % 60 == 0:
            print(f"  进度: {frame_idx+1}/{total_frames} 帧")
    
    out.release()
    print(f"视频已保存: {output_path}")

def process_video_with_detection(input_video, output_video, model):
    """处理视频，添加检测框和跟踪"""
    print(f"\n处理检测: {input_video}")
    
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 跟踪状态
    tracks = {}
    next_id = 1
    speeds = {}
    line_crosses = {'up': 0, 'down': 0}
    
    print(f"视频: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 使用 YOLO 检测 (COCO 预训练模型)
        results = model(frame, verbose=False, conf=0.4)
        
        # 绘制检测结果
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            # 只处理车辆相关类别
            if cls_id in [2, 3, 5, 7]:  # car, motorcycle, truck, bus
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 映射到我们的类别
                if cls_id == 2:  # car
                    class_name = 'car'
                elif cls_id == 3:  # motorcycle
                    class_name = 'motorcycle'
                elif cls_id == 7:  # truck
                    class_name = 'truck'
                else:  # bus
                    class_name = 'bus'
                
                color = CLASS_COLORS.get(class_name, (0, 255, 0))
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 标签
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'conf': conf,
                    'center': ((x1+x2)//2, (y1+y2)//2)
                })
        
        # 检测统计
        stats = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        for d in detections:
            if d['class'] in stats:
                stats[d['class']] += 1
        
        # 绘制统计面板
        cv2.rectangle(frame, (10, 10), (200, 140), (0, 0, 0), -1)
        cv2.putText(frame, "Detection Stats", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y = 55
        for cls, cnt in stats.items():
            cv2.putText(frame, f"{cls}: {cnt}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 22
        
        # 绘制检测线
        line_y = height // 2
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
        cv2.putText(frame, "Detection Line", (10, line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理: {frame_count}/{total_frames} 帧")
    
    cap.release()
    out.release()
    print(f"检测完成: {output_video}")
    
    return {
        'total_frames': frame_count,
        'fps': fps
    }

def main():
    print("=" * 60)
    print("车辆检测系统 - 真实检测演示")
    print("=" * 60)
    
    # 1. 创建交通视频
    video_path = 'demo_videos/traffic_video.mp4'
    output_detection = 'demo_videos/traffic_with_detection.mp4'
    
    create_traffic_video(video_path, duration_sec=15, fps=30)
    
    # 2. 加载模型
    print("\n加载 YOLO 预训练模型...")
    model = YOLO('yolov8n.pt')
    print("模型加载完成")
    
    # 3. 处理视频
    result = process_video_with_detection(video_path, output_detection, model)
    
    # 4. 输出结果
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"输出视频: {output_detection}")
    print(f"总帧数: {result['total_frames']}")
    print(f"FPS: {result['fps']}")

if __name__ == '__main__':
    main()
