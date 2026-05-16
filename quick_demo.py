#!/usr/bin/env python3
"""
快速生成演示视频
"""
import cv2
import numpy as np
import os

def create_quick_demo():
    """创建快速演示视频"""
    output_path = 'demo_videos/quick_demo.mp4'
    os.makedirs('demo_videos', exist_ok=True)
    
    width, height = 1280, 720
    fps = 30
    duration = 10  # 10秒
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    vehicles = [
        {'x': 100, 'y': 250, 'vx': 4, 'vy': 0.2, 'color': (0, 200, 0), 'type': 'car'},
        {'x': 400, 'y': 450, 'vx': 3, 'vy': -0.2, 'color': (255, 0, 0), 'type': 'truck'},
        {'x': 700, 'y': 350, 'vx': -3.5, 'vy': 0, 'color': (0, 165, 255), 'type': 'bus'},
    ]
    
    print(f'生成演示视频: {output_path}')
    print(f'分辨率: {width}x{height}, 时长: {duration}秒')
    
    for frame_idx in range(total_frames):
        # 背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # 道路
        cv2.rectangle(frame, (0, 180), (width, 540), (60, 60, 60), -1)
        
        # 车道线
        for y in range(180, 540, 50):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 2)
        
        # 中心线
        for x in range(0, width, 80):
            cv2.line(frame, (x, 355), (x + 40, 355), (0, 255, 255), 3)
        
        # 边框
        cv2.rectangle(frame, (0, 180), (width, 540), (255, 255, 255), 3)
        
        # 车辆
        for v in vehicles:
            v['x'] += v['vx']
            v['y'] += v['vy']
            
            if v['x'] > width + 100:
                v['x'] = -100
            if v['x'] < -100:
                v['x'] = width + 100
            if v['y'] > 480 or v['y'] < 230:
                v['vy'] = -v['vy']
            
            x, y = int(v['x']), int(v['y'])
            
            if v['type'] == 'car':
                cv2.rectangle(frame, (x, y), (x + 60, y + 35), v['color'], -1)
                cv2.rectangle(frame, (x + 10, y + 5), (x + 50, y + 20), (200, 200, 200), -1)
            elif v['type'] == 'truck':
                cv2.rectangle(frame, (x, y), (x + 90, y + 45), v['color'], -1)
                cv2.rectangle(frame, (x + 60, y + 5), (x + 85, y + 25), (200, 200, 200), -1)
            elif v['type'] == 'bus':
                cv2.rectangle(frame, (x, y), (x + 120, y + 40), v['color'], -1)
        
        # 标题
        cv2.putText(frame, 'Vehicle Detection Demo', (width//2 - 150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (frame_idx + 1) % fps == 0:
            print(f'  进度: {(frame_idx + 1) // fps}/{duration}秒')
    
    out.release()
    print(f'完成! 文件: {output_path}')

if __name__ == '__main__':
    create_quick_demo()
