"""
打包脚本
将系统打包为可执行文件
"""
import os
import sys
import shutil
import subprocess


def build_exe():
    """使用PyInstaller打包"""
    print("=" * 60)
    print("开始打包...")
    print("=" * 60)
    
    # 确保必要目录存在
    os.makedirs('weights', exist_ok=True)
    os.makedirs('demo_videos', exist_ok=True)
    
    # 复制必要的权重文件（如果没有）
    if not os.path.exists('weights/yolov8n.pt'):
        print("提示: 需要下载yolov8n.pt权重文件")
        print("运行: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
    
    # PyInstaller命令
    cmd = [
        'pyinstaller',
        '--name=VehicleDetectionSystem',
        '--onedir',  # 输出为目录
        '--windowed',  # 无控制台窗口
        '--icon=icon.ico' if os.path.exists('icon.ico') else '',
        '--add-data=models;models',
        '--add-data=weights;weights',
        '--add-data=demo_videos;demo_videos',
        '--add-data=data;data',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=ultralytics',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=PyQt5',
        '--hidden-import=deep_sort_realtime',
        '--collect-all=ultralytics',
        '--collect-all=torch',
        '--noconfirm',  # 覆盖已有文件
        'main.py'
    ]
    
    # 过滤空字符串
    cmd = [c for c in cmd if c]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n打包成功!")
        print("输出目录: dist/VehicleDetectionSystem")
        
        # 复制权重和演示视频到输出目录
        dist_dir = 'dist/VehicleDetectionSystem'
        if os.path.exists(dist_dir):
            # 创建目录
            os.makedirs(os.path.join(dist_dir, 'weights'), exist_ok=True)
            os.makedirs(os.path.join(dist_dir, 'demo_videos'), exist_ok=True)
            
            # 复制文件
            for f in os.listdir('weights'):
                src = os.path.join('weights', f)
                dst = os.path.join(dist_dir, 'weights', f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"复制: {f}")
            
            for f in os.listdir('demo_videos'):
                src = os.path.join('demo_videos', f)
                dst = os.path.join(dist_dir, 'demo_videos', f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"复制: {f}")
        
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")
    except FileNotFoundError:
        print("错误: PyInstaller未安装")
        print("请运行: pip install pyinstaller")


def create_installer():
    """创建安装包（Windows）"""
    print("创建安装包...")
    
    # 使用InnoSetup或其他工具
    # 这里只是提示
    print("提示: 可以使用以下工具创建安装包:")
    print("  - Inno Setup (Windows)")
    print("  - NSIS (Windows)")
    print("  - macOS: create-dmg (macOS)")
    print("  - Linux: fpm (Linux)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="打包脚本")
    parser.add_argument('--mode', type=str, default='exe',
                       choices=['exe', 'installer'],
                       help='打包模式')
    
    args = parser.parse_args()
    
    if args.mode == 'exe':
        build_exe()
    elif args.mode == 'installer':
        create_installer()
