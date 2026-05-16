"""
演示脚本
使用 COCO 预训练 YOLOv8 模型进行车辆检测
无需额外数据集，直接运行
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


# COCO 数据集中车辆相关类别
# YOLOv8 预训练模型自带这些类别的检测能力
CLASS_NAMES = {
    2: 'car',       # COCO: car
    3: 'motorcycle', # COCO: motorcycle
    5: 'bus',       # COCO: bus
    7: 'truck',     # COCO: truck
}

CLASS_COLORS = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 165, 255),
    'motorcycle': (255, 255, 0)
}

# 车辆类别的 COCO ID
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def load_model(model_path='yolov8n.pt', device='cpu'):
    """
    加载 YOLOv8 预训练模型
    
    Args:
        model_path: 模型路径，默认为 COCO 预训练的 yolov8n.pt
        device: 设备 (cpu/cuda)
        
    Returns:
        model: YOLO 模型
    """
    print(f"加载模型: {model_path}")
    
    # 检查本地是否有模型
    if not os.path.exists(model_path):
        print(f"本地模型不存在，尝试下载...")
        # 使用 COCO 预训练模型（内置于 ultralytics）
        model_path = 'yolov8n.pt'
    
    try:
        model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用 ultralytics 内置模型...")
        # ultralytics 会自动下载预训练权重
        return YOLO('yolov8n.pt')


def filter_vehicle_detections(results, conf_threshold=0.4):
    """
    从检测结果中筛选车辆
    
    Args:
        results: YOLO 检测结果
        conf_threshold: 置信度阈值
        
    Returns:
        detections: [[x1, y1, x2, y2, conf, class_id], ...]
    """
    detections = []
    
    if results is None or len(results) == 0:
        return detections
    
    result = results[0]
    if result.boxes is None:
        return detections
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    for i in range(len(boxes)):
        class_id = int(classes[i])
        conf = float(confs[i])
        
        # 只保留车辆类别
        if class_id in VEHICLE_CLASS_IDS and conf > conf_threshold:
            detections.append([
                boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                conf, class_id
            ])
    
    return detections


def draw_detection(frame, detection, track_id=None):
    """在图像上绘制检测框"""
    x1, y1, x2, y2, conf, class_id = detection
    class_name = CLASS_NAMES.get(class_id, 'unknown')
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    
    # 绘制边界框
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # 绘制标签
    label = f"{class_name}"
    if track_id is not None:
        label = f"ID:{track_id} {class_name}"
    label += f" {conf:.2f}"
    
    # 标签背景
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(x1), int(y1) - label_h - 5), 
                  (int(x1) + label_w, int(y1)), color, -1)
    
    # 标签文字
    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def process_video(video_path, model, show=True, conf_threshold=0.4):
    """
    处理视频
    
    Args:
        video_path: 视频文件路径
        model: YOLO 模型
        show: 是否显示窗口
        conf_threshold: 置信度阈值
    """
    print(f"\n处理视频: {video_path}")
    
    # 打开视频
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print("请运行以下命令生成演示视频:")
        print("  python create_demo_video.py")
        return
    
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
    
    # 初始化模块
    tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
    speed_estimator = SpeedEstimator(pixels_per_meter=30, fps=fps)
    line_counter = LineCounter(
        line_start=(0, height // 2), 
        line_end=(width, height // 2)
    )
    
    # 统计
    stats = {
        'total_detections': 0,
        'total_tracks': 0,
        'by_class': {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0},
        'crossed': 0
    }
    
    frame_idx = 0
    start_time = time.time()
    
    # 创建输出视频
    output_path = video_path.replace('.mp4', '_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        original_frame = frame.copy()
        
        # 检测
        results = model(frame, verbose=False)
        
        # 筛选车辆
        detections = filter_vehicle_detections(results, conf_threshold)
        stats['total_detections'] += len(detections)
        
        # 跟踪
        tracks = tracker.update(detections)
        stats['total_tracks'] = len(tracks)
        
        # 统计各类别
        for det in detections:
            class_id = int(det[5])
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            if class_name in stats['by_class']:
                stats['by_class'][class_name] += 1
        
        # 绘制检测结果
        for track in tracks:
            # track 格式: [x1, y1, x2, y2, track_id, class_id, ...]
            track_id = int(track[4]) if len(track) > 4 else 0
            class_id = int(track[5]) if len(track) > 5 else 0
            det = track[:6]
            
            # 更新速度
            speed_kmh = speed_estimator.update(
                track_id=track_id,
                bbox=det[:4],
                frame_idx=frame_idx
            )
            
            # 绘制
            draw_detection(frame, det, track_id)
            
            # 显示速度
            if speed_kmh > 0:
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                cv2.putText(frame, f"{speed_kmh:.1f} km/h", 
                           (cx - 30, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 检查跨线
            if len(det) >= 4:
                result = line_counter.check_cross(
                    track_id=track_id,
                    bbox=det[:4]
                )
                if result['crossed']:
                    stats['crossed'] += 1
        
        # 绘制检测线
        cv2.line(frame, 
                (int(line_counter.line_start[0]), int(line_counter.line_start[1])),
                (int(line_counter.line_end[0]), int(line_counter.line_end[1])),
                (0, 255, 255), 2)
        
        # 显示统计信息
        info_text = f"Frame: {frame_idx} | Detections: {len(detections)} | Tracks: {len(tracks)} | Crossed: {stats['crossed']}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 写入输出视频
        out.write(frame)
        
        # 显示窗口
        if show:
            cv2.imshow('Vehicle Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 释放资源
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    
    # 打印统计
    elapsed = time.time() - start_time
    print(f"\n处理完成!")
    print(f"总帧数: {frame_idx}")
    print(f"总检测数: {stats['total_detections']}")
    print(f"各类别检测数: {stats['by_class']}")
    print(f"跨线计数: {stats['crossed']}")
    print(f"处理时间: {elapsed:.2f}秒")
    print(f"平均FPS: {frame_idx / elapsed:.2f}")
    print(f"结果已保存: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="车辆检测与速度估算演示")
    parser.add_argument('--video', type=str, default='demo_videos/demo_traffic.mp4',
                       help='视频文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型路径 (默认为 COCO 预训练 yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='置信度阈值 (默认 0.4)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示窗口')
    args = parser.parse_args()
    
    print("=" * 60)
    print("基于改进 YOLOv8 的车辆检测与速度估算系统")
    print("=" * 60)
    print(f"使用 COCO 预训练模型")
    print(f"支持检测: car, truck, bus, motorcycle")
    print("=" * 60)
    
    # 加载模型
    model = load_model(args.model, args.device)
    
    # 处理视频
    process_video(args.video, model, show=not args.no_show, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
