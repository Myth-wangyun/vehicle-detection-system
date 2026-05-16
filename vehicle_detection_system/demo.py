"""
演示脚本
演示车辆检测与速度估算功能
"""
import os
import sys
import cv2
import numpy as np
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from tracker.deepsort_tracker import SimpleTracker
from tracker.speed_estimator import SpeedEstimator
from counter.line_counter import LineCounter


CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 165, 255),
    'motorcycle': (255, 255, 0)
}


def process_video(video_path, model, show=True):
    """处理视频"""
    print(f"处理视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
    
    # 初始化跟踪器和估算器
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    speed_estimator = SpeedEstimator(pixels_per_meter=30, fps=fps)
    line_counter = LineCounter(line_start=(0, height // 2), 
                               line_end=(width, height // 2))
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # 检测
        results = model(frame, verbose=False)
        
        # 提取检测结果
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                if confs[i] > 0.4:
                    detections.append([
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                        confs[i], int(classes[i])
                    ])
        
        # 跟踪
        tracks = tracker.update(detections)
        
        # 更新速度和计数
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            class_id = track.get('class_id', 0)
            
            speed = speed_estimator.update(track_id, bbox, frame_idx)
            line_counter.check_cross(track_id, bbox, speed, time.time())
            
            # 绘制
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown'
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id} {class_name}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制检测线
        start, end = line_counter.get_line()
        cv2.line(frame, tuple([int(x) for x in start]), 
                tuple([int(x) for x in end]), (0, 255, 255), 2)
        
        # 绘制统计信息
        cv2.rectangle(frame, (10, 10), (250, 120), (40, 40, 40), -1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Detections: {len(tracks)}", (15, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        counts = line_counter.get_count()
        cv2.putText(frame, f"Up: {counts['up']}  Down: {counts['down']}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 计算实际FPS
        elapsed = time.time() - start_time
        actual_fps = frame_idx / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示
        if show:
            cv2.imshow('Vehicle Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)  # 暂停
        
        # 打印进度
        if frame_idx % 100 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} ({frame_idx*100/total_frames:.1f}%)")
    
    cap.release()
    if show:
        cv2.destroyAllWindows()
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成!")
    print("=" * 50)
    print(f"总帧数: {frame_idx}")
    print(f"总时间: {time.time() - start_time:.2f}秒")
    print(f"平均FPS: {frame_idx / (time.time() - start_time):.2f}")
    print(f"上行计数: {counts['up']}")
    print(f"下行计数: {counts['down']}")


def main():
    """主函数"""
    # 加载模型
    print("加载模型...")
    
    # 优先使用CBAM改进模型
    if os.path.exists('weights/yolov8n_cbam.pt'):
        model = YOLO('weights/yolov8n_cbam.pt')
        print("使用CBAM改进模型")
    elif os.path.exists('weights/yolov8n_base.pt'):
        model = YOLO('weights/yolov8n_base.pt')
        print("使用基础YOLOv8模型")
    else:
        model = YOLO('yolov8n.pt')
        print("使用预训练模型 yolov8n.pt")
    
    # 查找演示视频
    demo_video = None
    video_dirs = ['demo_videos', '.', 'data']
    for d in video_dirs:
        for f in os.listdir(d) if os.path.exists(d) else []:
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                demo_video = os.path.join(d, f)
                break
    
    if demo_video:
        print(f"找到演示视频: {demo_video}")
        process_video(demo_video, model, show=True)
    else:
        print("未找到演示视频，请将视频文件放入 demo_videos 目录")
        
        # 使用摄像头测试
        print("尝试使用摄像头...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("摄像头已连接，开始实时检测...")
            process_video(0, model, show=True)
        else:
            print("无法打开摄像头")


if __name__ == "__main__":
    main()
