# AGENTS.md - 车辆检测系统项目规范

## 项目概述
基于改进YOLOv8的车辆速度与计数检测系统，实现车辆检测、多目标跟踪、速度估算和跨线计数功能。
论文目标：mAP@0.5 ≥ 92.4%，参数量 ≤ 5M，Jetson Nano 推理 ≥ 30 FPS。

## 技术栈
- **核心框架**: PyTorch 2.12.0, Ultralytics 8.4.51
- **跟踪算法**: DeepSORT, deep-sort-realtime 1.3.2
- **图像处理**: OpenCV 4.13.0, Pillow, SciPy
- **界面**: PyQt5 5.15.11
- **其他**: NumPy, Pandas, Matplotlib, Seaborn

## 目录结构
```
vehicle_detection_system/
├── main.py                 # 主入口 (PyQt5 GUI)
├── train.py                # 训练脚本
├── demo.py                 # 演示脚本
├── quick_demo.py           # 快速演示
├── create_demo_video.py    # 生成演示视频
├── build.py                # 打包脚本
├── prepare_dataset.py       # 数据集准备工具
├── requirements.txt        # 依赖列表
├── install.sh              # 依赖安装脚本
├── run.sh                  # 快速启动脚本
│
├── models/                 # 模型模块
│   ├── cbam.py            # CBAM注意力模块 ✅
│   ├── diou_loss.py        # DIoU损失函数 ✅
│   ├── register.py         # 模块注册
│   ├── yolov8n_cbam.yaml   # CBAM改进模型配置
│   └── yolov8n_lightweight.yaml  # 轻量化模型配置
│
├── tracker/               # 跟踪模块
│   ├── deepsort_tracker.py  # DeepSORT跟踪器 ✅
│   └── speed_estimator.py   # 速度估算器 (EWMA平滑) ✅
│
├── counter/               # 计数模块
│   └── line_counter.py      # 跨线计数器 ✅
│
├── ui/                    # 界面模块
│   └── main_window.py       # PyQt5主界面 ✅
│
├── utils/                 # 工具模块
│   ├── data_converter.py     # VOC/COCO转YOLO格式
│   └── augmentation.py       # Mosaic + MixUp 增强
│
├── weights/               # 模型权重目录
├── demo_videos/           # 演示视频目录 ✅
└── data/                  # 数据集配置
```

## 关键入口 / 核心模块
- **main.py**: PyQt5 GUI 主窗口入口
- **demo.py**: 命令行演示脚本
- **train.py**: 模型训练（支持 base/cbam/lightweight）
- **prepare_dataset.py**: 数据集下载和格式转换

## 运行与预览

### 预览链路（Web 预览）
- **预览页面**: `preview.html` - 炫酷可视化展示页面
- **预览服务**: Python HTTP 服务器，端口 5000
- **演示视频**: `demo_videos/simulation_output.mp4` - 模拟检测演示
- **访问地址**: Coze 平台预览区域

### 本地运行
```bash
# 安装依赖
bash install.sh

# 启动 GUI
python main.py

# 命令行演示（自动检测无头模式）
python demo.py --headless

# 模拟检测演示（展示完整功能）
python simulate_detection.py
```

## 实际情况说明

### 已完成
- ✅ 核心模块源码（CBAM、DIoU、DeepSORT、速度估算、跨线计数）
- ✅ PyQt5 GUI 界面
- ✅ 训练脚本（含数据增强）
- ✅ 模拟检测演示视频
- ✅ 炫酷可视化预览页面
- ✅ 依赖安装完成

### 待用户完成
- ⏳ 下载 UA-DETRAC 数据集（官方链接已失效，需自行寻找资源）
- ⏳ 实际训练（需 GPU 环境，训练时间 3-5 小时）
- ⏳ 论文数据验证（mAP 92.4% 等为目标值，非实测）

### 论文数据说明
当前展示的 mAP、参数量等数据为 **论文目标值**，非实际训练结果：
- mAP@0.5 ≥ 92.4%（论文目标）
- 参数量 ≤ 5M（轻量化目标）
- Jetson Nano ≥ 30 FPS（边缘部署目标）

真实数据需在本地 GPU 机器上训练后获取。

# 运行演示
python demo.py

# 下载 YOLO 预训练权重
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 生成演示视频
python create_demo_video.py
```

## 用户偏好与长期约束
- Python 版本要求 >= 3.8
- 依赖安装可使用国内镜像源加速
- GUI 应用，无 HTTP 服务端口要求
- 云环境无 GPU，建议用 CPU 训练或使用预训练模型

## 论文目标指标
| 模型 | mAP@0.5 | 参数量 | 推理速度 |
|------|---------|--------|----------|
| 原始 YOLOv8n | 85.6% | 3.2M | 18 FPS (Jetson) |
| 完整改进版 | 95.97% | 15.2M | - |
| 轻量化版 | 92.4% | 4.8M | 32 FPS (Jetson) |

## 常见问题和预防
- PyTorch 未安装：先运行 install.sh
- 模型权重缺失：首次运行会自动下载 yolov8n.pt
- 无演示视频：运行 quick_demo.py 或 create_demo_video.py
- 打包失败：确保 weights 和 demo_videos 目录存在

## 交付清单
- [x] 完整源码
- [ ] 训练好的权重（需自行训练）
- [ ] UA-DETRAC 数据集（需手动下载）
- [ ] 对比实验数据表格（需训练后生成）
- [x] .exe 可执行文件（需运行 pyinstaller）
