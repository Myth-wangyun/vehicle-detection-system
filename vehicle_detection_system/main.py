"""
主入口文件
基于改进YOLOv8的车辆速度与计数检测系统
使用 COCO 预训练模型，无需额外数据集
"""
import sys
import os
import argparse

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于改进YOLOv8的车辆速度与计数检测系统")
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='模型路径 (默认为 COCO 预训练 yolov8n.pt)')
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
    
    # 检查依赖
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
    except ImportError:
        print("错误: PyQt5 未安装")
        print("请运行: pip install PyQt5")
        return
    
    # 尝试导入 UI
    try:
        from ui.main_window import MainWindow
    except Exception as e:
        print(f"UI 导入失败: {e}")
        print("尝试运行演示模式...")
        # 如果 UI 导入失败，运行命令行演示
        import subprocess
        cmd = [sys.executable, 'demo.py']
        if args.video:
            cmd.extend(['--video', args.video])
        subprocess.run(cmd)
        return
    
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
