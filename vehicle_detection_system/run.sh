#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "车辆检测与速度检测系统"
echo "=========================================="

# 检查依赖
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: PyTorch未安装"
    echo "请先运行: bash install.sh"
    exit 1
fi

python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: ultralytics未安装"
    echo "请先运行: bash install.sh"
    exit 1
fi

# 检查模型权重
if [ ! -f "yolov8n.pt" ] && [ ! -f "weights/yolov8n_cbam.pt" ]; then
    echo "下载YOLOv8预训练权重..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# 检查演示视频
if [ ! -f "demo_videos/quick_demo.mp4" ]; then
    echo "生成演示视频..."
    python3 quick_demo.py
fi

# 运行主程序
echo ""
echo "启动主程序..."
python3 main.py
