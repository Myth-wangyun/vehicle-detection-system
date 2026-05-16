"""
模型训练脚本
训练YOLOv8及其改进版本
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


def train_base_model(data_config='data/vehicle.yaml', epochs=80, batch=4, imgsz=640, device='cpu'):
    """
    训练基础YOLOv8模型
    
    Args:
        data_config: 数据集配置文件
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 图像尺寸
        device: 设备
    """
    print("=" * 60)
    print("训练基础YOLOv8模型")
    print("=" * 60)
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project='runs/train',
        name='yolov8n_base',
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
        val=True,
    )
    
    # 保存权重
    best_path = f"runs/train/yolov8n_base/weights/best.pt"
    if os.path.exists(best_path):
        os.makedirs('weights', exist_ok=True)
        import shutil
        shutil.copy(best_path, 'weights/yolov8n_base.pt')
        print(f"权重已保存到: weights/yolov8n_base.pt")
    
    return results


def train_cbam_model(data_config='data/vehicle.yaml', epochs=80, batch=4, imgsz=640, device='cpu'):
    """
    训练YOLOv8n-CBAM改进模型
    
    Args:
        data_config: 数据集配置文件
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 图像尺寸
        device: 设备
    """
    print("=" * 60)
    print("训练YOLOv8n-CBAM改进模型")
    print("=" * 60)
    
    # 检查模型配置文件
    yaml_path = 'models/yolov8n_cbam.yaml'
    if not os.path.exists(yaml_path):
        print(f"警告: 模型配置文件 {yaml_path} 不存在，跳过CBAM模型训练")
        return None
    
    try:
        # 加载自定义模型
        model = YOLO(yaml_path)
        
        # 尝试加载预训练权重
        if os.path.exists('yolov8n.pt'):
            model = YOLO(yaml_path).load('yolov8n.pt')
        
        # 开始训练
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project='runs/train',
            name='yolov8n_cbam',
            exist_ok=True,
            verbose=True,
            save=True,
            plots=True,
            val=True,
        )
        
        # 保存权重
        best_path = f"runs/train/yolov8n_cbam/weights/best.pt"
        if os.path.exists(best_path):
            os.makedirs('weights', exist_ok=True)
            import shutil
            shutil.copy(best_path, 'weights/yolov8n_cbam.pt')
            print(f"CBAM权重已保存到: weights/yolov8n_cbam.pt")
        
        return results
        
    except Exception as e:
        print(f"CBAM模型训练失败: {str(e)}")
        return None


def train_lightweight_model(data_config='data/vehicle.yaml', epochs=80, batch=4, imgsz=640, device='cpu'):
    """
    训练轻量化模型
    
    Args:
        data_config: 数据集配置文件
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 图像尺寸
        device: 设备
    """
    print("=" * 60)
    print("训练轻量化YOLOv8模型")
    print("=" * 60)
    
    yaml_path = 'models/yolov8n_lightweight.yaml'
    if not os.path.exists(yaml_path):
        print(f"警告: 模型配置文件 {yaml_path} 不存在，跳过轻量化模型训练")
        return None
    
    try:
        model = YOLO(yaml_path)
        if os.path.exists('yolov8n.pt'):
            model = YOLO(yaml_path).load('yolov8n.pt')
        
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project='runs/train',
            name='yolov8n_light',
            exist_ok=True,
            verbose=True,
            save=True,
            plots=True,
            val=True,
        )
        
        # 保存权重
        best_path = f"runs/train/yolov8n_light/weights/best.pt"
        if os.path.exists(best_path):
            os.makedirs('weights', exist_ok=True)
            import shutil
            shutil.copy(best_path, 'weights/yolov8n_lightweight.pt')
            print(f"轻量化权重已保存到: weights/yolov8n_lightweight.pt")
        
        return results
        
    except Exception as e:
        print(f"轻量化模型训练失败: {str(e)}")
        return None


def train_all(data_config='data/vehicle.yaml', epochs=80, batch=4, imgsz=640, device='cpu'):
    """训练所有模型"""
    print("\n" + "=" * 60)
    print("开始训练所有模型")
    print("=" * 60)
    print(f"数据集配置: {data_config}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch}")
    print(f"图像尺寸: {imgsz}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs('weights', exist_ok=True)
    os.makedirs('runs/train', exist_ok=True)
    
    # 训练基础模型
    print("\n[1/3] 训练基础YOLOv8模型...")
    train_base_model(data_config, epochs, batch, imgsz, device)
    
    # 训练CBAM模型
    print("\n[2/3] 训练CBAM改进模型...")
    train_cbam_model(data_config, epochs, batch, imgsz, device)
    
    # 训练轻量化模型
    print("\n[3/3] 训练轻量化模型...")
    train_lightweight_model(data_config, epochs, batch, imgsz, device)
    
    print("\n" + "=" * 60)
    print("所有模型训练完成!")
    print("=" * 60)
    
    # 列出已保存的权重
    print("\n已保存的权重文件:")
    for f in os.listdir('weights'):
        print(f"  - weights/{f}")


def export_model(weights_path, format='onnx'):
    """导出模型为其他格式"""
    print(f"导出模型: {weights_path} -> {format}")
    
    if not os.path.exists(weights_path):
        print(f"权重文件不存在: {weights_path}")
        return
    
    model = YOLO(weights_path)
    model.export(format=format)
    
    print(f"导出完成!")


def compare_models(weights_list):
    """比较不同模型的性能"""
    print("=" * 60)
    print("模型性能比较")
    print("=" * 60)
    
    for weights in weights_list:
        if not os.path.exists(weights):
            continue
        
        print(f"\n模型: {weights}")
        model = YOLO(weights)
        
        # 获取模型信息
        print(f"  参数量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
        
        # 简单测试
        import numpy as np
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        import time
        start = time.time()
        for _ in range(10):
            model(dummy_input)
        elapsed = (time.time() - start) / 10
        
        print(f"  平均推理时间: {elapsed * 1000:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8训练脚本")
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'export', 'compare'],
                       help='运行模式')
    parser.add_argument('--data', type=str, default='data/vehicle.yaml',
                       help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=80,
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='图像尺寸')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['base', 'cbam', 'light', 'all'],
                       help='模型类型')
    parser.add_argument('--weights', type=str, default=None,
                       help='模型权重路径')
    parser.add_argument('--format', type=str, default='onnx',
                       help='导出格式')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.model_type == 'all':
            train_all(args.data, args.epochs, args.batch, args.imgsz, args.device)
        elif args.model_type == 'base':
            train_base_model(args.data, args.epochs, args.batch, args.imgsz, args.device)
        elif args.model_type == 'cbam':
            train_cbam_model(args.data, args.epochs, args.batch, args.imgsz, args.device)
        elif args.model_type == 'light':
            train_lightweight_model(args.data, args.epochs, args.batch, args.imgsz, args.device)
    
    elif args.mode == 'export':
        if args.weights:
            export_model(args.weights, args.format)
        else:
            print("请指定要导出的权重文件: --weights")
    
    elif args.mode == 'compare':
        weights_list = [
            'weights/yolov8n_base.pt',
            'weights/yolov8n_cbam.pt',
            'weights/yolov8n_lightweight.pt'
        ]
        compare_models(weights_list)
