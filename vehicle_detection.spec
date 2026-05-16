# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 添加数据文件
datas = [
    ('models/*.yaml', 'models'),
    ('models/*.py', 'models'),
    ('weights/*.pt', 'weights'),
    ('weights/*.onnx', 'weights'),
    ('demo_videos/*.mp4', 'demo_videos'),
    ('data/*.yaml', 'data'),
]

# 隐藏导入
hiddenimports = [
    'torch',
    'torchvision',
    'ultralytics',
    'cv2',
    'numpy',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'deep_sort_realtime',
    'scipy',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VehicleDetectionSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 无控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VehicleDetectionSystem',
)

# 创建Windows安装包（可选）
# from cx_Freeze import setup, Executable
# setup(...)
