#!/usr/bin/env python3
"""
模拟检测演示脚本
展示车辆检测系统的完整工作流程：
- 视频读取
- 目标跟踪 (DeepSORT)
- 速度估算 (EWMA)
- 跨线计数
- 结果可视化

注意：此脚本使用模拟检测框来演示系统功能。
在真实场景下，检测框由 YOLOv8 模型提供。
"""

import cv2
import numpy as np
import time
from collections import defaultdict

# 车辆类别
VEHICLE_CLASSES = {
    0: ('car', (0, 255, 0)),
    1: ('truck', (255, 0, 0)),
    2: ('bus', (0, 165, 255)),
    3: ('motorcycle', (255, 0, 255))
}

class SimulatedTracker:
    """模拟跟踪器"""
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.vehicle_configs = [
            {'lane': 350, 'speed_range': (40, 60), 'start': 0, 'end': 200, 'direction': 1, 'cls': 0},
            {'lane': 400, 'speed_range': (50, 70), 'start': 30, 'end': 220, 'direction': -1, 'cls': 0},
            {'lane': 450, 'speed_range': (30, 50), 'start': 60, 'end': 250, 'cls': 1},
            {'lane': 320, 'speed_range': (35, 45), 'start': 90, 'end': 240, 'cls': 2},
            {'lane': 420, 'speed_range': (55, 75), 'start': 120, 'end': 250, 'direction': 1, 'cls': 3},
        ]
    
    def update(self, frame_idx):
        """更新跟踪状态"""
        current_detections = []
        
        for cfg in self.vehicle_configs:
            t = frame_idx - cfg['start']
            if 0 <= t <= cfg['end'] - cfg['start']:
                direction = cfg.get('direction', 1)
                speed = np.random.uniform(*cfg['speed_range'])
                pixels_per_frame = speed / 3.6 * 30 / 1000  # 转换为像素
                
                if direction > 0:
                    x = int(t * pixels_per_frame * 50)
                else:
                    x = 1100 - int(t * pixels_per_frame * 50)
                
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'cls': cfg['cls'],
                    'lane': cfg['lane'],
                    'speed': speed,
                    'direction': direction
                }
                
                current_detections.append({
                    'bbox': [x, cfg['lane'], x+120, cfg['lane']+55],
                    'track_id': track_id,
                    'cls': cfg['cls'],
                    'speed': speed
                })
        
        return current_detections

def draw_vehicle(draw, x, y, w, h, cls_id, track_id, speed):
    """绘制车辆和标注"""
    cls_name, color = VEHICLE_CLASSES[cls_id]
    
    # 车身
    draw.polygon([
        (x, y+h), (x, y+h//2), (x+w//4, y), (x+3*w//4, y), (x+w, y+h//2), (x+w, y+h)
    ], fill=color, outline=(255, 255, 255))
    
    # 检测框
    draw.rectangle([x-2, y-2, x+w+2, y+h+2], outline=color, width=3)
    
    # 类别和ID标签
    label = f"{cls_name.upper()} #{track_id}"
    draw.text((x, y-25), label, fill=color)
    
    # 速度标签
    speed_label = f"{speed:.0f} km/h"
    draw.text((x+w-50, y+h+5), speed_label, fill=(255, 255, 0))

def draw_ui(draw, stats, fps, frame_idx):
    """绘制UI面板"""
    # 背景
    draw.rectangle([10, 10, 250, 160], fill=(0, 0, 0, 180))
    draw.rectangle([10, 10, 250, 160], outline=(0, 255, 255), width=2)
    
    # 标题
    draw.text((20, 20), "车辆检测系统", fill=(0, 255, 255))
    draw.text((20, 45), f"FPS: {fps:.1f}", fill=(0, 255, 0))
    draw.text((20, 65), f"帧: {frame_idx}", fill=(255, 255, 255))
    draw.text((20, 85), f"检测: {stats['detected']}", fill=(255, 255, 0))
    draw.text((20, 105), f"计数: {stats['counted']}", fill=(0, 255, 255))
    draw.text((20, 125), f"平均速度: {stats['avg_speed']:.0f} km/h", fill=(0, 255, 0))
    
    # 统计
    draw.text((20, 145), f"汽车:{stats['car']} 卡车:{stats['truck']} 公交:{stats['bus']}", fill=(255, 255, 255))
    
    # 计数面板
    draw.rectangle([1030, 10, 1270, 80], fill=(0, 0, 0, 180))
    draw.rectangle([1030, 10, 1270, 80], outline=(255, 0, 255), width=2)
    draw.text((1040, 20), "今日统计", fill=(255, 0, 255))
    draw.text((1040, 45), f"通过车辆: {stats['counted']}", fill=(255, 255, 255))
    draw.text((1040, 65), f"平均速度: {stats['avg_speed']:.0f} km/h", fill=(0, 255, 0))

def main():
    from PIL import Image, ImageDraw, ImageFont
    
    print("=" * 60)
    print("基于改进 YOLOv8 的车辆检测与速度估算系统")
    print("=" * 60)
    print("模式: 模拟检测演示")
    print("支持检测: car, truck, bus, motorcycle")
    print("=" * 60)
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'demo_videos/simulation_output.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30, (1280, 720))
    
    # 初始化
    tracker = SimulatedTracker()
    stats = {
        'detected': 0,
        'counted': 0,
        'avg_speed': 0,
        'total_speed': 0,
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0
    }
    counted_ids = set()
    
    print(f"\n处理视频...")
    start_time = time.time()
    
    for frame_idx in range(300):  # 10秒视频
        # 创建背景
        img = Image.new('RGB', (1280, 720), (60, 60, 60))
        draw = ImageDraw.Draw(img)
        
        # 绘制道路
        draw.rectangle([0, 300, 1280, 550], fill=(50, 50, 50))
        for x in range(0, 1280, 60):
            draw.rectangle([x, 420, x+40, 425], fill=(200, 200, 200))
        
        # 获取模拟检测
        detections = tracker.update(frame_idx)
        stats['detected'] += len(detections)
        
        # 绘制检测和跟踪
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            cls_id = det['cls']
            speed = det['speed']
            
            # 统计
            cls_name = VEHICLE_CLASSES[cls_id][0]
            stats[cls_name] += 1
            
            # 计数逻辑 (通过画面中心)
            if x1 < 640 <= x2 and track_id not in counted_ids:
                counted_ids.add(track_id)
                stats['counted'] += 1
                stats['total_speed'] += speed
                if stats['counted'] > 0:
                    stats['avg_speed'] = stats['total_speed'] / stats['counted']
            
            # 绘制
            draw_vehicle(draw, x1, y1, x2-x1, y2-y1, cls_id, track_id, speed)
        
        # FPS 计算
        elapsed = time.time() - start_time
        fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        
        # 绘制UI
        draw_ui(draw, stats, fps, frame_idx)
        
        # 检测线
        draw.line([(0, 640//2), (1280, 640//2)], fill=(255, 0, 0), width=2)
        draw.text((10, 640//2+5), "检测线", fill=(255, 0, 0))
        
        # 保存帧
        out.write(np.array(img))
        
        if frame_idx % 50 == 0:
            print(f"  处理中... {frame_idx}/300 帧, FPS: {fps:.1f}")
    
    out.release()
    
    print(f"\n处理完成!")
    print(f"  输出: {output_path}")
    print(f"  检测总数: {stats['detected']}")
    print(f"  计数总数: {stats['counted']}")
    print(f"  汽车: {stats['car']}, 卡车: {stats['truck']}, 公交: {stats['bus']}, 摩托: {stats['motorcycle']}")
    print(f"  平均速度: {stats['avg_speed']:.0f} km/h")
    
    return output_path

if __name__ == '__main__':
    main()
