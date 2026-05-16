#!/bin/bash
# 安装脚本 - 自动安装所有依赖

echo "=========================================="
echo "车辆检测系统 - 依赖安装"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建虚拟环境（可选）
# python3 -m venv venv
# source venv/bin/activate

# 使用清华源安装
echo ""
echo "正在安装依赖..."
echo ""

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    ultralytics \
    torch \
    torchvision \
    opencv-python \
    opencv-python-headless \
    PyQt5 \
    deep-sort-realtime \
    numpy \
    pandas \
    scipy \
    Pillow \
    PyYAML \
    matplotlib \
    seaborn \
    tqdm \
    pyinstaller

echo ""
echo "=========================================="
echo "依赖安装完成!"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 下载YOLOv8预训练权重 (如果需要):"
echo "   python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\""
echo ""
echo "2. 生成演示视频:"
echo "   python quick_demo.py"
echo ""
echo "3. 运行演示程序:"
echo "   python main.py"
echo ""
echo "4. 训练模型:"
echo "   python train.py --model-type all --epochs 80 --batch 4 --device cpu"
echo ""
