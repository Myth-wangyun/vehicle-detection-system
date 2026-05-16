#!/usr/bin/env python3
"""检查依赖安装状态"""
print("="*50)
print("依赖安装状态检查")
print("="*50)

deps = [
    ("torch", "PyTorch"),
    ("cv2", "OpenCV"),
    ("ultralytics", "Ultralytics"),
    ("PyQt5", "PyQt5"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
]

for module, name in deps:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError:
        print(f"✗ {name}: 未安装")

print("="*50)
