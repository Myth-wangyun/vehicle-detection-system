"""
主入口文件
基于改进YOLOv8的车辆速度与计数检测系统
"""
import sys
import os
import argparse

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于改进YOLOv8的车辆速度与计数检测系统")
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='模型路径')
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头ID')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
