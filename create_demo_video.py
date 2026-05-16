"""
生成演示视频
使用OpenCV合成模拟车辆行驶的视频
"""
import cv2
import numpy as np
import os


def create_demo_video(output_path='demo_videos/demo_traffic.mp4', duration=30, fps=30):
    """
    创建演示视频
    
    Args:
        output_path: 输出路径
        duration: 视频时长（秒）
        fps: 帧率
    """
    print(f"生成演示视频: {output_path}")
    
    width, height = 1280, 720
    total_frames = duration * fps
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 车辆参数
    vehicles = [
        {'x': 100, 'y': 200, 'vx': 3, 'vy': 0.5, 'color': (0, 200, 0), 'type': 'car'},
        {'x': 200, 'y': 400, 'vx': 4, 'vy': -0.3, 'color': (255, 0, 0), 'type': 'truck'},
        {'x': 50, 'y': 500, 'vx': 2.5, 'vy': 0.2, 'color': (0, 165, 255), 'type': 'bus'},
        {'x': 600, 'y': 300, 'vx': -3, 'vy': 0, 'color': (255, 255, 0), 'type': 'moto'},
        {'x': 800, 'y': 150, 'vx': 3.5, 'vy': 0.1, 'color': (0, 255, 0), 'type': 'car'},
    ]
    
    for frame_idx in range(total_frames):
        # 创建背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # 绘制道路
        cv2.rectangle(frame, (0, 150), (width, 570), (60, 60, 60), -1)
        
        # 绘制车道线
        for y in range(150, 570, 40):
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 2)
        
        # 绘制中心虚线
        for x in range(0, width, 60):
            cv2.line(frame, (x, 355), (x + 30, 355), (255, 255, 0), 3)
        
        # 绘制边框
        cv2.rectangle(frame, (0, 150), (width, 570), (255, 255, 255), 3)
        
        # 添加路边建筑
        for i in range(5):
            bx = i * 300 + (frame_idx * 0.5) % 300
            if bx > width:
                bx -= width
            cv2.rectangle(frame, (int(bx), 30), (int(bx) + 80, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (int(bx), 570), (int(bx) + 80, 690), (100, 100, 100), -1)
        
        # 绘制车辆
        for v in vehicles:
            # 更新位置
            v['x'] += v['vx']
            v['y'] += v['vy']
            
            # 边界检测
            if v['x'] > width + 100:
                v['x'] = -100
            if v['x'] < -100:
                v['x'] = width + 100
            if v['y'] > 520 or v['y'] < 200:
                v['vy'] = -v['vy']
            
            # 绘制车辆
            x, y = int(v['x']), int(v['y'])
            
            if v['type'] == 'car':
                cv2.rectangle(frame, (x, y), (x + 60, y + 35), v['color'], -1)
                cv2.rectangle(frame, (x + 10, y + 5), (x + 50, y + 20), (200, 200, 200), -1)
            elif v['type'] == 'truck':
                cv2.rectangle(frame, (x, y), (x + 90, y + 45), v['color'], -1)
                cv2.rectangle(frame, (x + 60, y + 5), (x + 85, y + 25), (200, 200, 200), -1)
            elif v['type'] == 'bus':
                cv2.rectangle(frame, (x, y), (x + 120, y + 40), v['color'], -1)
                for i in range(4):
                    cv2.rectangle(frame, (x + 10 + i * 28, y + 5), (x + 35 + i * 28, y + 25), (200, 200, 200), -1)
            elif v['type'] == 'moto':
                cv2.circle(frame, (x + 10, y + 15), 8, v['color'], -1)
                cv2.circle(frame, (x + 30, y + 15), 8, v['color'], -1)
                cv2.rectangle(frame, (x + 5, y), (x + 35, y + 10), v['color'], -1)
        
        # 添加标题
        cv2.putText(frame, f"Demo Traffic Video - Frame {frame_idx}/{total_frames}", 
                   (width // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加水印
        cv2.putText(frame, "Qilu University of Technology", 
                   (width - 250, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 写入帧
        out.write(frame)
        
        # 打印进度
        if (frame_idx + 1) % fps == 0:
            print(f"进度: {frame_idx // fps + 1}/{duration}秒")
    
    out.release()
    print(f"演示视频生成完成: {output_path}")


def create_detection_demo():
    """创建检测演示"""
    print("创建检测演示...")
    
    # 确保目录存在
    os.makedirs('demo_videos', exist_ok=True)
    
    # 生成视频
    create_demo_video('demo_videos/demo_traffic.mp4', duration=20, fps=30)


if __name__ == "__main__":
    create_detection_demo()
