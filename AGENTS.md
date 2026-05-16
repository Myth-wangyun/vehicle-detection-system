# AGENTS.md - 车辆检测系统项目规范

## 项目概述
基于改进YOLOv8的车辆速度与计数检测系统，实现车辆检测、多目标跟踪、速度估算和跨线计数功能。

## 技术栈
- **核心框架**: PyTorch, Ultralytics (YOLOv8)
- **跟踪算法**: DeepSORT, deep-sort-realtime
- **图像处理**: OpenCV, Pillow, SciPy
- **界面**: PyQt5
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
├── requirements.txt        # 依赖列表
├── install.sh              # 依赖安装脚本
├── run.sh                  # 快速启动脚本
│
├── models/                 # 模型模块
│   ├── cbam.py            # CBAM注意力模块
│   ├── diou_loss.py       # DIoU损失函数
│   └── yolov8n_*.yaml     # 模型配置
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
├── weights/               # 模型权重目录
├── demo_videos/           # 演示视频目录
└── data/                  # 数据集配置
```

## 关键入口 / 核心模块
- **main.py**: PyQt5 GUI 主窗口入口
- **demo.py**: 命令行演示脚本
- **quick_demo.py**: 快速演示
- **train.py**: 模型训练

## 运行与预览
```bash
# 安装依赖
bash install.sh

# 启动 GUI
python main.py

# 或使用快捷脚本
bash run.sh

# 运行演示
python demo.py
```

## 用户偏好与长期约束
- Python 版本要求 >= 3.8
- 依赖安装可使用国内镜像源加速
- GUI 应用，无 HTTP 服务端口要求

## 常见问题和预防
- PyTorch 未安装：先运行 install.sh
- 模型权重缺失：首次运行会自动下载 yolov8n.pt
- 无演示视频：运行 quick_demo.py 生成
