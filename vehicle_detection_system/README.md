# 基于改进YOLOv8的车辆速度与计数检测系统

齐鲁工业大学 本科毕业论文演示系统

## 项目概述

本项目实现了一个完整的车辆检测与速度估算系统，主要特点：

1. **改进的YOLOv8模型**：集成CBAM注意力机制和DIoU损失函数
2. **DeepSORT多目标跟踪**：实现车辆ID跟踪和轨迹记录
3. **速度估算**：基于帧间位移和像素比例系数的速度计算
4. **跨线计数**：虚拟检测线 + 方向判别 + 防重复计数
5. **模型轻量化**：深度可分离卷积替换
6. **PyQt5可视化界面**：视频/摄像头切换、实时检测、统计面板

## 项目结构

```
vehicle_detection_system/
├── main.py                 # 主入口
├── train.py                # 训练脚本
├── demo.py                 # 演示脚本
├── create_demo_video.py    # 生成演示视频
├── build.py                # 打包脚本
├── requirements.txt        # 依赖列表
│
├── models/                 # 模型模块
│   ├── cbam.py            # CBAM注意力模块
│   ├── diou_loss.py       # DIoU损失函数
│   ├── yolov8n_cbam.yaml  # CBAM改进模型配置
│   └── yolov8n_lightweight.yaml  # 轻量化模型配置
│
├── tracker/               # 跟踪模块
│   ├── deepsort_tracker.py  # DeepSORT跟踪器
│   └── speed_estimator.py   # 速度估算器
│
├── counter/               # 计数模块
│   └── line_counter.py      # 跨线计数器
│
├── ui/                    # 界面模块
│   └── main_window.py       # PyQt5主界面
│
├── utils/                 # 工具模块
│   ├── data_converter.py    # 数据格式转换
│   └── augmentation.py      # 数据增强
│
├── weights/               # 模型权重目录
├── demo_videos/           # 演示视频目录
└── data/                  # 数据集配置
```

## 环境配置

### 基本依赖

```bash
pip install -r requirements.txt
```

或使用国内源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 主要依赖

- Python >= 3.8
- PyTorch >= 2.0
- Ultralytics (YOLOv8)
- OpenCV
- PyQt5
- deep-sort-realtime

## 使用方法

### 1. 运行演示界面

```bash
cd vehicle_detection_system
python main.py
```

### 2. 运行演示脚本

```bash
python demo.py
```

### 3. 训练模型

```bash
# 训练基础YOLOv8模型
python train.py --model-type base --epochs 80 --batch 4

# 训练CBAM改进模型
python train.py --model-type cbam --epochs 80 --batch 4

# 训练所有模型
python train.py --model-type all --epochs 80 --batch 4
```

### 4. 生成演示视频

```bash
python create_demo_video.py
```

### 5. 打包

```bash
python build.py --mode exe
```

## 功能说明

### 车辆检测
- 支持4种车型：轿车(car)、卡车(truck)、公交车(bus)、摩托车(motorcycle)
- 基于YOLOv8n的轻量级检测模型
- 可调节置信度阈值

### 速度估算
- 基于帧间位移计算瞬时速度
- EWMA平滑算法减少抖动
- 像素比例系数可配置

### 跨线计数
- 虚拟检测线设置
- 自动判断车辆行驶方向（上/下/左/右）
- 防重复计数机制

### 界面功能
- 视频文件/摄像头输入切换
- 模型切换（原始/CBAM改进/轻量化）
- 实时统计面板
- 数据导出（CSV）
- 截图保存

## 模型说明

### CBAM注意力机制
通道注意力 + 空间注意力的串行组合，增强特征表达

### DIoU损失函数
考虑边界框中心距离的IoU损失，加速收敛

### 深度可分离卷积
轻量化模型设计，减少参数量和计算量

## 注意事项

1. CPU训练较慢，建议使用GPU或减少训练轮数
2. 首次运行会自动下载YOLOv8预训练权重
3. 速度估算需要根据实际场景校准像素/米比例
4. 打包前确保所有权重文件已放置在weights目录

## 许可证

MIT License

## 作者

齐鲁工业大学
