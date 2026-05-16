#!/bin/bash
# ============================================
# 车辆检测系统 - 一键启动
# ============================================

cd "$(dirname "$0")"

echo "=========================================="
echo "基于改进YOLOv8的车辆检测与速度估算系统"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查依赖
echo "[1/4] 检查依赖..."
python3 -c "import torch, ultralytics, cv2, PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "请先安装依赖: bash install.sh"
    exit 1
fi
echo "  依赖检查通过"

# 检查视频文件
echo "[2/4] 检查演示文件..."
if [ ! -f "demo_videos/demo.mp4" ]; then
    echo "  生成演示视频..."
    python3 generate_complete_system.py
fi
echo "  演示文件就绪"

# 启动预览服务
echo "[3/4] 启动预览服务..."
pkill -f "http.server 5000" 2>/dev/null
nohup python3 -m http.server 5000 > /tmp/preview.log 2>&1 &
sleep 1

# 检查预览服务
if curl -s --max-time 2 http://localhost:5000/preview.html > /dev/null; then
    echo "  预览服务已启动: http://localhost:5000/preview.html"
else
    echo "  警告: 预览服务启动失败"
fi

# 显示选项
echo "[4/4] 就绪!"
echo ""
echo "=========================================="
echo "启动选项:"
echo "=========================================="
echo ""
echo "1. 打开炫酷可视化页面 (推荐):"
echo "   浏览器访问: http://localhost:5000/preview.html"
echo ""
echo "2. 命令行演示 (无GUI):"
echo "   python demo.py --video demo_videos/demo.mp4 --headless"
echo ""
echo "3. GUI界面 (需要图形环境):"
echo "   python main.py"
echo ""
echo "4. 查看生成的视频:"
echo "   ls demo_videos/"
echo ""
echo "=========================================="
echo ""

# 如果有图形界面，启动GUI
if [ -z "$DISPLAY" ]; then
    echo "无图形界面，自动进入命令行模式..."
    python3 demo.py --video demo_videos/demo.mp4 --headless
else
    echo "检测到图形环境，可以运行 GUI 版本"
fi
