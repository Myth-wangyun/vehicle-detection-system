"""
生成更真实的演示视频
使用 OpenCV 绘制更逼真的车辆
"""
import cv2
import numpy as np
import os


def draw_car(frame, x, y, width=80, height=50, color=(0, 200, 0)):
    """绘制汽车"""
    # 车身
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
    # 车窗
    cv2.rectangle(frame, (x + 10, y + 5), (x + width - 10, y + height // 2), (100, 150, 200), -1)
    # 车轮
    cv2.circle(frame, (x + 15, y + height), 10, (30, 30, 30), -1)
    cv2.circle(frame, (x + width - 15, y + height), 10, (30, 30, 30), -1)
    return frame


def draw_truck(frame, x, y, width=100, height=60, color=(255, 0, 0)):
    """绘制卡车"""
    # 车身
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
    # 车厢
    cv2.rectangle(frame, (x + 30, y + 5), (x + width - 5, y + height - 5), (100, 100, 100), -1)
    # 车轮
    cv2.circle(frame, (x + 20, y + height), 12, (30, 30, 30), -1)
    cv2.circle(frame, (x + width - 20, y + height), 12, (30, 30, 30), -1)
    return frame


def draw_bus(frame, x, y, width=120, height=50, color=(0, 165, 255)):
    """绘制公交车"""
    # 车身
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
    # 窗户
    for wx in range(x + 10, x + width - 10, 15):
        cv2.rectangle(frame, (wx, y + 5), (wx + 10, y + height // 2), (150, 200, 220), -1)
    # 车轮
    cv2.circle(frame, (x + 20, y + height), 12, (30, 30, 30), -1)
    cv2.circle(frame, (x + width - 20, y + height), 12, (30, 30, 30), -1)
    return frame


def draw_motorcycle(frame, x, y, width=40, height=30, color=(255, 255, 0)):
    """绘制摩托车"""
    # 车身
    cv2.ellipse(frame, (x + width // 2, y + height // 2), (width // 2, height // 3), 0, 0, 360, color, -1)
    # 车轮
    cv2.circle(frame, (x + 5, y + height // 2), 8, (30, 30, 30), -1)
    cv2.circle(frame, (x + width - 5, y + height // 2), 8, (30, 30, 30), -1)
    return frame


def create_realistic_video(output_path='demo_videos/realistic_traffic.mp4', duration=20, fps=30):
    """创建更真实的演示视频"""
    print(f"生成真实感演示视频: {output_path}")

    width, height = 1280, 720
    total_frames = duration * fps

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 车辆配置
    vehicles = [
        {'type': 'car', 'x': 100, 'y': 200, 'vx': 5, 'vy': 0, 'color': (0, 180, 0), 'w': 80, 'h': 45},
        {'type': 'truck', 'x': 300, 'y': 320, 'vx': 4, 'vy': 0, 'color': (200, 50, 50), 'w': 100, 'h': 55},
        {'type': 'bus', 'x': 500, 'y': 450, 'vx': 3.5, 'vy': 0, 'color': (50, 150, 255), 'w': 120, 'h': 45},
        {'type': 'car', 'x': 700, 'y': 200, 'vx': 6, 'vy': 0, 'color': (100, 100, 200), 'w': 75, 'h': 42},
        {'type': 'motorcycle', 'x': 200, 'y': 550, 'vx': 7, 'vy': 0, 'color': (255, 200, 0), 'w': 35, 'h': 25},
        {'type': 'car', 'x': 900, 'y': 380, 'vx': -4, 'vy': 0, 'color': (150, 200, 150), 'w': 78, 'h': 44},
        {'type': 'truck', 'x': 1000, 'y': 520, 'vx': -3, 'vy': 0, 'color': (180, 80, 80), 'w': 95, 'h': 52},
    ]

    for frame_idx in range(total_frames):
        # 天空背景
        frame = np.ones((height, width, 3), dtype=np.uint8)
        frame[:, :] = (135, 206, 235)  # 浅蓝色天空

        # 远处背景
        cv2.rectangle(frame, (0, 120), (width, 200), (100, 150, 100), -1)  # 草地

        # 道路
        cv2.rectangle(frame, (0, 180), (width, 600), (80, 80, 80), -1)  # 道路

        # 道路边缘线
        cv2.line(frame, (0, 180), (width, 180), (255, 255, 255), 3)
        cv2.line(frame, (0, 600), (width, 600), (255, 255, 255), 3)

        # 车道线
        for i in range(3):
            lane_y = 300 + i * 100
            for x in range(0, width, 80):
                cv2.line(frame, (x, lane_y), (x + 40, lane_y), (255, 255, 255), 2)

        # 更新和绘制车辆
        for v in vehicles:
            v['x'] += v['vx']

            # 超出边界则重置
            if v['x'] > width + 200:
                v['x'] = -200
            elif v['x'] < -200:
                v['x'] = width + 200

            # 根据类型绘制
            if v['type'] == 'car':
                draw_car(frame, int(v['x']), int(v['y']), v['w'], v['h'], v['color'])
            elif v['type'] == 'truck':
                draw_truck(frame, int(v['x']), int(v['y']), v['w'], v['h'], v['color'])
            elif v['type'] == 'bus':
                draw_bus(frame, int(v['x']), int(v['y']), v['w'], v['h'], v['color'])
            elif v['type'] == 'motorcycle':
                draw_motorcycle(frame, int(v['x']), int(v['y']), v['w'], v['h'], v['color'])

        # 添加帧号
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        if frame_idx % 100 == 0:
            print(f"进度: {frame_idx}/{total_frames} 帧")

    out.release()
    print(f"视频生成完成: {output_path}")
    print(f"分辨率: {width}x{height}, 时长: {duration}秒, FPS: {fps}")


if __name__ == "__main__":
    create_realistic_video()
