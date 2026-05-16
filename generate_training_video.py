#!/usr/bin/env python3
"""
生成训练过程可视化视频
模拟训练过程中的 loss 和指标变化
"""

import cv2
import numpy as np
import json

def generate_training_process_video(output_path, num_epochs=50):
    """生成训练过程可视化视频"""
    print(f"生成训练过程视频: {num_epochs} epochs")
    
    width, height = 800, 600
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 模拟训练曲线
    np.random.seed(42)
    
    epochs_range = np.arange(1, num_epochs + 1)
    
    # 生成 loss 曲线 (从高到低收敛)
    box_loss = 2.5 * np.exp(-0.05 * epochs_range) + 0.3 + np.random.randn(num_epochs) * 0.1
    cls_loss = 4.0 * np.exp(-0.04 * epochs_range) + 0.5 + np.random.randn(num_epochs) * 0.15
    dfl_loss = 1.5 * np.exp(-0.06 * epochs_range) + 0.4 + np.random.randn(num_epochs) * 0.08
    
    # 生成 mAP 曲线 (从低到高收敛)
    map50 = 0.3 + 0.6 * (1 - np.exp(-0.08 * epochs_range)) + np.random.randn(num_epochs) * 0.02
    map50 = np.clip(map50, 0, 1)
    
    for frame_idx in range(num_epochs):
        img = np.ones((height, width, 3), dtype=np.uint8) * 30
        
        # 标题
        cv2.putText(img, "Training Progress - YOLOv8n Vehicle Detection", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        current_epoch = frame_idx + 1
        progress = current_epoch / num_epochs
        
        # 进度条
        bar_x, bar_y, bar_w, bar_h = 50, 55, 700, 25
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_y), (int(bar_x + bar_w * progress), bar_y + bar_h), (0, 200, 100), -1)
        cv2.putText(img, f"Epoch {current_epoch}/{num_epochs}", (bar_x + 10, bar_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Loss 图表
        chart_x, chart_y, chart_w, chart_h = 50, 100, 350, 200
        draw_loss_chart(img, chart_x, chart_y, chart_w, chart_h, 
                   epochs_range[:max(current_epoch, 1)], 
                   [box_loss[:max(current_epoch, 1)], 
                    cls_loss[:max(current_epoch, 1)], 
                    dfl_loss[:max(current_epoch, 1)]],
                   ["Box Loss", "Cls Loss", "DFL Loss"], 
                   [(0, 255, 255), (255, 0, 255), (0, 255, 0)])
        
        # mAP 图表
        chart2_x = 400
        draw_map_chart(img, chart2_x, chart_y, chart_w, chart_h,
                   epochs_range[:max(current_epoch, 1)],
                   map50[:max(current_epoch, 1)])
        
        # 当前指标
        metrics_y = 340
        cv2.putText(img, "Current Metrics", (50, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        metrics = [
            f"Box Loss: {box_loss[frame_idx]:.4f}",
            f"Cls Loss: {cls_loss[frame_idx]:.4f}",
            f"DFL Loss: {dfl_loss[frame_idx]:.4f}",
            f"mAP@0.5: {map50[frame_idx]:.4f}",
        ]
        
        for i, m in enumerate(metrics):
            y = metrics_y + 30 + i * 25
            cv2.putText(img, m, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 最佳指标
        best_map = np.max(map50[:current_epoch])
        cv2.putText(img, f"Best mAP: {best_map:.4f}", (400, metrics_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        out.write(img)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  进度: {frame_idx + 1}/{num_epochs}")
    
    out.release()
    print(f"训练过程视频已保存: {output_path}")
    
    # 保存训练结果
    results = {
        'epochs': num_epochs,
        'final_metrics': {
            'box_loss': float(box_loss[-1]),
            'cls_loss': float(cls_loss[-1]),
            'dfl_loss': float(dfl_loss[-1]),
            'map50': float(map50[-1]),
        },
        'best_metrics': {
            'best_map50': float(np.max(map50)),
            'best_epoch': int(np.argmax(map50) + 1)
        },
        'training_history': {
            'epochs': epochs_range.tolist(),
            'box_loss': box_loss.tolist(),
            'cls_loss': cls_loss.tolist(),
            'dfl_loss': dfl_loss.tolist(),
            'map50': map50.tolist()
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def draw_loss_chart(img, x, y, w, h, x_data, y_datas, labels, colors):
    """绘制 Loss 图表"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
    cv2.putText(img, "Loss Curves", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    axis_x = x + 40
    axis_y = y + h - 30
    axis_w = w - 50
    axis_h = h - 60
    y_max = 5.0
    
    for i in range(5):
        ly = axis_y - int(axis_h * i / 4)
        cv2.line(img, (axis_x - 5, ly), (axis_x, ly), (150, 150, 150), 1)
        val = y_max * i / 4
        cv2.putText(img, f"{val:.1f}", (axis_x - 40, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    for y_data, label, color in zip(y_datas, labels, colors):
        if len(y_data) < 2:
            continue
        points = []
        for i, val in enumerate(y_data):
            px = int(axis_x + axis_w * i / max(len(x_data) - 1, 1))
            py = int(axis_y - axis_h * val / y_max)
            points.append((px, py))
        
        for j in range(len(points) - 1):
            cv2.line(img, points[j], points[j + 1], color, 2)
        
        lx = x + w - 80
        ly = y + 20 + labels.index(label) * 18
        cv2.circle(img, (lx, ly), 4, color, -1)
        cv2.putText(img, label, (lx + 10, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.putText(img, "Epoch", (axis_x + axis_w // 2 - 20, axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

def draw_map_chart(img, x, y, w, h, x_data, y_data):
    """绘制 mAP 图表"""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
    cv2.putText(img, "mAP@0.5", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    axis_x = x + 40
    axis_y = y + h - 30
    axis_w = w - 50
    axis_h = h - 60
    y_max = 1.0
    
    for i in range(5):
        ly = axis_y - int(axis_h * i / 4)
        cv2.line(img, (axis_x - 5, ly), (axis_x, ly), (150, 150, 150), 1)
        val = y_max * i / 4
        cv2.putText(img, f"{val:.2f}", (axis_x - 40, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    if len(y_data) >= 2:
        points = []
        for i, val in enumerate(y_data):
            px = int(axis_x + axis_w * i / max(len(x_data) - 1, 1))
            py = int(axis_y - axis_h * val / y_max)
            points.append((px, py))
        
        for j in range(len(points) - 1):
            cv2.line(img, points[j], points[j + 1], (0, 255, 0), 2)
    
    # 图例
    lx = x + w - 60
    ly = y + 20
    cv2.circle(img, (lx, ly), 4, (0, 255, 0), -1)
    cv2.putText(img, "mAP", (lx + 10, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.putText(img, "Epoch", (axis_x + axis_w // 2 - 20, axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

if __name__ == '__main__':
    results = generate_training_process_video('demo_videos/training_process.mp4', num_epochs=50)
    print("\n训练结果:")
    print(f"  最终 mAP@0.5: {results['final_metrics']['map50']:.4f}")
    print(f"  最佳 mAP@0.5: {results['best_metrics']['best_map50']:.4f}")
