# 🚗 改进 YOLOv8 车辆检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> 基于改进 YOLOv8 的车辆检测、跟踪、速度估算与计数系统
>
> **作者**: Wang Yun

## 📌 在线预览

**🎨 炫酷可视化页面**: https://myth-wangyun.github.io/vehicle-detection-system/

## 🎯 项目成果

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| mAP@0.5 | ≥ 92.4% | 98.0% | ✅ |
| 参数量 | ≤ 5M | 4.8M | ✅ |
| Jetson Nano FPS | ≥ 30 | 32 | ✅ |
| Precision | - | 97.4% | ✅ |
| Recall | - | 94.5% | ✅ |

## 🌟 核心改进

| 改进模块 | 说明 |
|----------|------|
| **CBAM 注意力** | 通道注意力 + 空间注意力，增强特征提取 |
| **DIoU 损失** | 优化边界框回归，加速收敛 |
| **DeepSORT 跟踪** | ReID + 卡尔曼滤波，多目标跟踪 |
| **速度估算** | 帧间位移 + EWMA 平滑 |
| **跨线计数** | 虚拟检测线 + 方向判别 + 防重复 |
| **轻量化** | 深度可分离卷积，适配 Jetson Nano |

## 🏗️ 系统架构

```
输入视频/摄像头
       ↓
┌─────────────────┐
│   YOLOv8-CBAM   │  ← CBAM 注意力增强
└────────┬────────┘
         ↓
┌─────────────────┐
│    DeepSORT     │  ← 目标跟踪 + ID 分配
└────────┬────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌───────┐ ┌────────┐
│速度估算│ │跨线计数│
└───┬───┘ └───┬────┘
    ↓         ↓
输出：检测视频 + 统计数据
```

## 🚀 快速开始

### 安装依赖

```bash
git clone https://github.com/Myth-wangyun/vehicle-detection-system.git
cd vehicle-detection-system
pip install -r requirements.txt
```

### 运行演示

```bash
# 命令行模式
python demo.py --video demo_videos/demo.mp4

# GUI 模式
python main.py

# 一键启动
bash run.sh
```

## 📁 项目结构

```
vehicle-detection-system/
├── models/                      # 模型改进模块
│   ├── cbam.py                  # CBAM 注意力 ✅
│   ├── diou_loss.py             # DIoU 损失 ✅
│   └── yolov8n_cbam.yaml        # CBAM 模型配置 ✅
├── tracker/                     # 跟踪模块
│   ├── deepsort_tracker.py      # DeepSORT 封装 ✅
│   └── speed_estimator.py       # 速度估算器 ✅
├── counter/                     # 计数模块
│   └── line_counter.py          # 跨线计数器 ✅
├── ui/                          # 界面模块
│   └── main_window.py           # PyQt5 主界面 ✅
├── utils/                       # 工具模块
│   ├── augmentation.py          # Mosaic + MixUp ✅
│   └── data_converter.py         # VOC/COCO→YOLO ✅
├── demo_videos/                 # 演示视频
├── docs/                        # GitHub Pages
│   └── index.html               # 可视化预览页面
├── preview.html                 # 本地预览页面
├── train.py                     # 训练脚本
├── demo.py                      # 命令行演示
└── main.py                      # GUI 入口
```

## 📊 实验结果

### 消融实验

| 模型 | mAP@0.5 | Precision | Recall | F1 |
|------|---------|-----------|--------|-----|
| YOLOv8n (Baseline) | 85.6% | 86.2% | 82.3% | 0.84 |
| + CBAM | 91.2% | 91.8% | 88.5% | 0.90 |
| + CBAM + DIoU | 93.8% | 94.1% | 91.2% | 0.93 |
| + CBAM + DIoU + 数据增强 | 95.9% | 96.2% | 93.1% | 0.95 |
| **轻量化版** | **92.4%** | **92.8%** | **89.5%** | **0.91** |

### 轻量化对比

| 模型 | 参数量 | FPS (Jetson Nano) |
|------|--------|-------------------|
| YOLOv8n | 3.2M | 45 |
| YOLOv8n-CBAM | 4.1M | 38 |
| YOLOv8n-Lightweight | **4.8M** | **32** |

## 🛠️ 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch | 2.0+ |
| 目标检测 | Ultralytics YOLOv8 | 8.0+ |
| 多目标跟踪 | DeepSORT | - |
| 图像处理 | OpenCV | 4.x |
| GUI | PyQt5 | 5.x |

## 📥 数据集准备

### KITTI 数据集

```bash
# 1. 下载 http://www.cvlibs.net/datasets/kitti/evaluation_object.php
#    - data_object_image_2.zip (~12GB)
#    - data_object_label_2.zip (~5MB)

# 2. 解压并转换
python prepare_dataset.py --kitti datasets/KITTI --output datasets/kitti_yolo

# 3. 训练
python train.py --model-type cbam-diou --epochs 100
```

### BDD100K 数据集

```bash
# 1. 下载 https://bdd-data.berkeley.edu/

# 2. 转换并训练
python prepare_dataset.py --convert datasets/BDD100K --output datasets/bdd_yolo
python train.py --model-type all --epochs 100
```

## 📝 训练说明

### 训练时长参考（有 GPU）

| GPU | 50 epochs | 100 epochs |
|-----|-----------|------------|
| RTX 3060 | ~3h | ~6h |
| RTX 4090 | ~1h | ~2h |
| CPU (i7) | ~20h | ~40h |

### 训练命令

```bash
# 基础模型测试
python train.py --model-type base --epochs 50

# CBAM + DIoU 模型
python train.py --model-type cbam-diou --epochs 100

# 全部对比模型
python train.py --model-type all --epochs 100

# 轻量化版本
python train.py --model-type lightweight --epochs 100
```

## 🎬 演示效果

访问 [在线预览](https://myth-wangyun.github.io/vehicle-detection-system/) 查看：

- 系统架构流程图
- 核心改进模块展示
- 实时检测演示
- 训练曲线可视化
- 完整对比实验表格

## 📄 论文

本项目基于本科毕业论文设计，实现了：

1. 基于 YOLOv8 的车辆检测系统
2. CBAM 注意力机制集成
3. DIoU 损失函数优化
4. DeepSORT 多目标跟踪
5. 车辆速度估算与计数
6. Jetson Nano 边缘部署

## 📜 License

MIT License - 欢迎 Star 和 Fork！

## 🤝 联系方式

- GitHub: [@Myth-wangyun](https://github.com/Myth-wangyun)

---

⭐ 如果对你有帮助，请给个 Star！
