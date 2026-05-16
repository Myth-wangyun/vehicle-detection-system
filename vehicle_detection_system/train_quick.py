#!/usr/bin/env python3
"""
快速训练脚本 - 使用合成数据集训练 YOLOv8
专为快速生成可演示的结果而设计
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import json
import time

def train_model(model_name, data_config, epochs=30, device='cpu'):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 加载预训练模型
    if model_name == 'yolov8n':
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO('yolov8n.pt')  # 使用预训练模型
    
    # 训练
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=4,
        imgsz=640,
        device=device,
        project='runs/train',
        name=model_name.replace('.', '_'),
        exist_ok=True,
        verbose=False,  # 减少输出
        save=True,
        plots=False,  # 跳过绘图加速训练
        val=True,
        workers=0,  # 减少内存使用
        cache=False,
    )
    
    elapsed = time.time() - start_time
    
    # 获取结果
    metrics = results.results_dict if hasattr(results, 'results_dict') else {}
    
    return {
        'model': model_name,
        'epochs': epochs,
        'time': elapsed,
        'metrics': metrics
    }

def main():
    print("=" * 60)
    print("YOLOv8 车辆检测模型训练")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    data_config = 'data/synthetic.yaml'
    
    # 检查数据
    if not os.path.exists(data_config):
        print(f"错误: 数据配置文件 {data_config} 不存在")
        return
    
    dataset_path = 'datasets/synthetic'
    train_images = len(list(Path(f'{dataset_path}/images/train').glob('*.jpg')))
    val_images = len(list(Path(f'{dataset_path}/images/val').glob('*.jpg')))
    print(f"\n数据集:")
    print(f"  训练集: {train_images} 张")
    print(f"  验证集: {val_images} 张")
    
    # 确定设备
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # 训练配置
    epochs = 50 if torch.cuda.is_available() else 20
    
    print(f"\n开始训练 (device={device}, epochs={epochs})...")
    
    # 训练基础模型
    result = train_model('yolov8n_vehicle', data_config, epochs=epochs, device=device)
    
    # 保存结果
    os.makedirs('weights', exist_ok=True)
    
    # 复制权重
    best_weights = 'runs/train/yolov8n_vehicle/weights/best.pt'
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, 'weights/yolov8n_vehicle_best.pt')
        print(f"\n权重已保存: weights/yolov8n_vehicle_best.pt")
    
    # 生成训练结果报告
    report = {
        'training_completed': True,
        'model': 'yolov8n',
        'dataset': 'synthetic (800 images)',
        'epochs': epochs,
        'device': device,
        'results': result
    }
    
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n训练报告已保存: training_report.json")
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
