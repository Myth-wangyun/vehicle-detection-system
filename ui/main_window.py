"""
PyQt5 主界面
基于改进YOLOv8的车辆速度与计数检测系统
"""
import sys
import os
import time
import threading
import numpy as np
import cv2
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QGroupBox, QFileDialog,
    QMessageBox, QTextEdit, QProgressBar, QCheckBox, QFrame,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QSplitter, QStyleFactory, QApplication
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.deepsort_tracker import DeepSortTracker, SimpleTracker
from tracker.speed_estimator import SpeedEstimator
from counter.line_counter import LineCounter


# 类别名称
CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
CLASS_COLORS = {
    'car': (0, 255, 0),        # 绿色
    'truck': (255, 0, 0),      # 蓝色
    'bus': (0, 165, 255),      # 橙色
    'motorcycle': (255, 255, 0) # 黄色
}


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(np.ndarray)
    stats_update = pyqtSignal(dict)
    detection_update = pyqtSignal(list)
    
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.running = False
        self.paused = False
        self.cap = None
        self.video_path = None
        
        # 跟踪器和速度估算器
        self.tracker = SimpleTracker(max_age=30, iou_threshold=0.3)
        self.speed_estimator = SpeedEstimator(pixels_per_meter=30, fps=30)
        self.line_counter = None
        
        # 统计信息
        self.stats = {
            'total_count': 0,
            'car_count': 0,
            'truck_count': 0,
            'bus_count': 0,
            'motorcycle_count': 0,
            'avg_speed': 0,
            'current_fps': 0,
            'frame_count': 0
        }
        
        # 速度分布
        self.speed_bins = [0, 20, 40, 60, 80, 100, 120]
        self.speed_distribution = {i: 0 for i in range(len(self.speed_bins) - 1)}
        
        # 计数
        self.count_up = 0
        self.count_down = 0
        
        # FPS计算
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
    
    def set_video(self, path):
        """设置视频路径"""
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.reset()
    
    def set_camera(self, camera_id=0):
        """设置摄像头"""
        self.cap = cv2.VideoCapture(camera_id)
        self.reset()
    
    def set_detection_line(self, start, end):
        """设置检测线"""
        self.line_counter = LineCounter(line_start=start, line_end=end)
    
    def reset(self):
        """重置状态"""
        self.stats = {
            'total_count': 0,
            'car_count': 0,
            'truck_count': 0,
            'bus_count': 0,
            'motorcycle_count': 0,
            'avg_speed': 0,
            'current_fps': 0,
            'frame_count': 0
        }
        self.speed_distribution = {i: 0 for i in range(len(self.speed_bins) - 1)}
        self.count_up = 0
        self.count_down = 0
        self.speed_estimator.reset()
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def run(self):
        """处理视频帧"""
        self.running = True
        self.paused = False
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            # 处理帧
            processed_frame, detections, track_info = self.process_frame(frame)
            
            # 更新FPS
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            self.stats['current_fps'] = self.current_fps
            self.stats['frame_count'] += 1
            
            # 发送信号
            self.frame_ready.emit(processed_frame)
            self.stats_update.emit(self.stats.copy())
            self.detection_update.emit(track_info)
            
            # 控制帧率
            time.sleep(0.03)  # ~30fps
    
    def process_frame(self, frame):
        """处理单帧"""
        # 检测
        results = self.model(frame, verbose=False)
        
        # 提取检测结果
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                if confs[i] > 0.3:  # 置信度阈值
                    detections.append([
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                        confs[i], int(classes[i])
                    ])
        
        # 跟踪
        tracks = self.tracker.update(detections)
        
        # 更新速度和计数
        track_info = []
        speeds = []
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            class_id = track.get('class_id', 0)
            
            # 速度估算
            speed = self.speed_estimator.update(
                track_id, bbox, self.stats['frame_count']
            )
            speeds.append(speed)
            
            # 检测线计数
            if self.line_counter is not None:
                self.line_counter.check_cross(
                    track_id, bbox, speed, time.time()
                )
            
            track_info.append({
                'id': track_id,
                'bbox': bbox,
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown',
                'speed': speed
            })
            
            # 更新统计
            self.stats['total_count'] = len(tracks)
            if class_id == 0:
                self.stats['car_count'] += 1
            elif class_id == 1:
                self.stats['truck_count'] += 1
            elif class_id == 2:
                self.stats['bus_count'] += 1
            elif class_id == 3:
                self.stats['motorcycle_count'] += 1
        
        # 计算平均速度
        if len(speeds) > 0:
            self.stats['avg_speed'] = np.mean([s for s in speeds if s > 0])
        
        # 获取计数
        if self.line_counter is not None:
            counts = self.line_counter.get_count()
            self.count_up = counts['up']
            self.count_down = counts['down']
        
        # 绘制结果
        annotated_frame = self.draw_annotations(frame.copy(), track_info)
        
        return annotated_frame, detections, track_info
    
    def draw_annotations(self, frame, track_info):
        """绘制标注"""
        h, w = frame.shape[:2]
        
        # 绘制检测线
        if self.line_counter is not None:
            start, end = self.line_counter.get_line()
            cv2.line(frame, tuple([int(x) for x in start]), 
                    tuple([int(x) for x in end]), (0, 255, 255), 2)
        
        # 绘制跟踪结果
        for info in track_info:
            bbox = info['bbox']
            track_id = info['id']
            class_name = info['class_name']
            speed = info['speed']
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # 边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"ID:{track_id} {class_name}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制速度分布图
        self.draw_speed_chart(frame)
        
        # 绘制计数信息
        self.draw_count_info(frame)
        
        return frame
    
    def draw_speed_chart(self, frame):
        """绘制速度分布图"""
        h, w = frame.shape[:2]
        chart_x, chart_y = w - 180, 30
        chart_w, chart_h = 150, 100
        
        # 背景
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), 
                     (40, 40, 40), -1)
        cv2.putText(frame, "Speed Distribution", (chart_x + 10, chart_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制柱状图
        max_count = max(max(self.speed_distribution.values()), 1)
        bar_width = (chart_w - 20) // len(self.speed_distribution)
        
        for i, (bin_idx, count) in enumerate(self.speed_distribution.items()):
            bar_height = int((count / max_count) * (chart_h - 30))
            x = chart_x + 10 + i * bar_width
            y = chart_y + chart_h - bar_height - 15
            
            cv2.rectangle(frame, (x, y), (x + bar_width - 2, chart_y + chart_h - 15), 
                         (0, 200, 100), -1)
        
        # 标注
        cv2.putText(frame, f"Avg: {self.stats['avg_speed']:.1f} km/h", 
                   (chart_x + 10, chart_y + chart_h + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def draw_count_info(self, frame):
        """绘制计数信息"""
        h, w = frame.shape[:2]
        
        # 背景
        cv2.rectangle(frame, (10, 10), (200, 120), (40, 40, 40), -1)
        
        # 标题
        cv2.putText(frame, "Detection Statistics", (15, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # 计数
        y = 50
        cv2.putText(frame, f"Total: {self.stats['total_count']}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        y += 18
        cv2.putText(frame, f"Cars: {self.stats['car_count']}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLORS['car'], 1)
        
        y += 16
        cv2.putText(frame, f"Trucks: {self.stats['truck_count']}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLORS['truck'], 1)
        
        y += 16
        cv2.putText(frame, f"Buses: {self.stats['bus_count']}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLORS['bus'], 1)
        
        y += 16
        cv2.putText(frame, f"Motorcycles: {self.stats['motorcycle_count']}", (15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLORS['motorcycle'], 1)
        
        # 跨线计数
        if self.line_counter is not None:
            cv2.putText(frame, f"Up: {self.count_up}  Down: {self.count_down}", 
                       (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    def pause(self):
        """暂停"""
        self.paused = True
    
    def resume(self):
        """继续"""
        self.paused = False
    
    def stop(self):
        """停止"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
    
    def is_running(self):
        """检查是否运行中"""
        return self.running


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口
        self.setWindowTitle("基于改进YOLOv8的车辆速度与计数检测系统")
        self.setGeometry(100, 100, 1400, 800)
        
        # 加载模型
        self.model = None
        self.video_thread = None
        
        # 当前状态
        self.current_source = None
        self.confidence_threshold = 0.4
        self.show_detection_line = True
        
        # 初始化UI
        self.init_ui()
        
        # 加载模型
        self.load_model()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        
        # 中间视频区域
        center_panel = self.create_video_panel()
        
        # 右侧统计面板
        right_panel = self.create_stats_panel()
        
        # 添加强制器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("控制面板")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 输入源选择
        source_group = QGroupBox("输入源")
        source_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["视频文件", "摄像头"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_combo)
        
        self.file_btn = QPushButton("选择视频文件")
        self.file_btn.clicked.connect(self.select_video_file)
        source_layout.addWidget(self.file_btn)
        
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setPrefix("摄像头: ")
        self.camera_spin.setVisible(False)
        source_layout.addWidget(self.camera_spin)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "YOLOv8n (原始)",
            "YOLOv8n-CBAM (改进)",
            "YOLOv8n-Light (轻量)"
        ])
        model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout()
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(40)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.40")
        conf_layout.addWidget(self.conf_label)
        param_layout.addLayout(conf_layout)
        
        # 检测线开关
        self.line_check = QCheckBox("显示检测线")
        self.line_check.setChecked(True)
        self.line_check.stateChanged.connect(self.on_line_toggled)
        param_layout.addWidget(self.line_check)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.start_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("padding: 8px;")
        self.pause_btn.clicked.connect(self.pause_detection)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 导出按钮
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout()
        
        self.export_csv_btn = QPushButton("导出CSV数据")
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        self.screenshot_btn = QPushButton("保存截图")
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        export_layout.addWidget(self.screenshot_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def create_video_panel(self):
        """创建视频显示面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("视频显示")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 视频标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("请选择视频文件或摄像头")
        layout.addWidget(self.video_label)
        
        # FPS显示
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.fps_label)
        
        return panel
    
    def create_stats_panel(self):
        """创建统计面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("检测统计")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 总体统计
        stats_group = QGroupBox("总体统计")
        stats_layout = QVBoxLayout()
        
        self.total_label = QLabel("检测总数: 0")
        self.total_label.setFont(QFont("Microsoft YaHei", 10))
        stats_layout.addWidget(self.total_label)
        
        self.avg_speed_label = QLabel("平均速度: 0.0 km/h")
        self.avg_speed_label.setFont(QFont("Microsoft YaHei", 10))
        stats_layout.addWidget(self.avg_speed_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # 车型统计
        vehicle_group = QGroupBox("车型统计")
        vehicle_layout = QVBoxLayout()
        
        self.car_label = QLabel("轿车: 0")
        self.car_label.setStyleSheet("color: green;")
        vehicle_layout.addWidget(self.car_label)
        
        self.truck_label = QLabel("卡车: 0")
        self.truck_label.setStyleSheet("color: blue;")
        vehicle_layout.addWidget(self.truck_label)
        
        self.bus_label = QLabel("公交车: 0")
        self.bus_label.setStyleSheet("color: orange;")
        vehicle_layout.addWidget(self.bus_label)
        
        self.moto_label = QLabel("摩托车: 0")
        self.moto_label.setStyleSheet("color: yellow;")
        vehicle_layout.addWidget(self.moto_label)
        
        vehicle_group.setLayout(vehicle_layout)
        layout.addWidget(vehicle_group)
        
        # 计数统计
        count_group = QGroupBox("跨线计数")
        count_layout = QVBoxLayout()
        
        self.up_count_label = QLabel("上行: 0")
        count_layout.addWidget(self.up_count_label)
        
        self.down_count_label = QLabel("下行: 0")
        count_layout.addWidget(self.down_count_label)
        
        count_group.setLayout(count_layout)
        layout.addWidget(count_group)
        
        # 速度分布表
        dist_group = QGroupBox("速度分布")
        dist_layout = QVBoxLayout()
        
        self.dist_table = QTableWidget()
        self.dist_table.setColumnCount(2)
        self.dist_table.setRowCount(6)
        self.dist_table.setHorizontalHeaderLabels(["速度区间", "数量"])
        self.dist_table.setEditable(False)
        
        ranges = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
        for i, r in enumerate(ranges):
            self.dist_table.setItem(i, 0, QTableWidgetItem(r))
            self.dist_table.setItem(i, 1, QTableWidgetItem("0"))
        
        dist_layout.addWidget(self.dist_table)
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)
        
        # 检测日志
        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def load_model(self):
        """加载YOLOv8模型"""
        self.statusBar().showMessage("正在加载模型...")
        
        try:
            from ultralytics import YOLO
            
            # 获取模型路径
            model_type = self.model_combo.currentIndex()
            if model_type == 0:
                model_path = "yolov8n.pt"
            elif model_type == 1:
                model_path = "weights/yolov8n_cbam.pt"
            else:
                model_path = "weights/yolov8n_lightweight.pt"
            
            # 尝试加载模型
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.log_text.append(f"已加载模型: {model_path}")
            else:
                # 使用预训练模型
                self.model = YOLO("yolov8n.pt")
                self.log_text.append("使用预训练模型: yolov8n.pt")
            
            self.statusBar().showMessage("模型加载完成")
            
        except Exception as e:
            self.statusBar().showMessage(f"模型加载失败: {str(e)}")
            self.log_text.append(f"错误: {str(e)}")
    
    def select_video_file(self):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", 
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_btn.setText(f"已选择: {os.path.basename(file_path)}")
            self.log_text.append(f"已选择视频: {file_path}")
    
    def on_source_changed(self, index):
        """输入源改变"""
        if index == 0:  # 视频文件
            self.file_btn.setVisible(True)
            self.camera_spin.setVisible(False)
        else:  # 摄像头
            self.file_btn.setVisible(False)
            self.camera_spin.setVisible(True)
    
    def on_confidence_changed(self, value):
        """置信度改变"""
        self.confidence_threshold = value / 100.0
        self.conf_label.setText(f"{self.confidence_threshold:.2f}")
    
    def on_line_toggled(self, state):
        """检测线开关"""
        self.show_detection_line = state == Qt.Checked
    
    def start_detection(self):
        """开始检测"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载!")
            return
        
        # 获取输入源
        source_type = self.source_combo.currentIndex()
        
        # 创建视频线程
        self.video_thread = VideoThread(self.model)
        
        # 设置输入源
        if source_type == 0:  # 视频文件
            if hasattr(self, 'video_path'):
                self.video_thread.set_video(self.video_path)
            else:
                QMessageBox.warning(self, "警告", "请先选择视频文件!")
                return
        else:  # 摄像头
            self.video_thread.set_camera(self.camera_spin.value())
        
        # 设置检测线
        if self.show_detection_line:
            # 默认检测线位置
            self.video_thread.set_detection_line(
                start=(0, 360),
                end=(1280, 360)
            )
        
        # 连接信号
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.stats_update.connect(self.update_stats)
        self.video_thread.detection_update.connect(self.update_detections)
        
        # 启动线程
        self.video_thread.start()
        
        # 更新按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.statusBar().showMessage("检测进行中...")
        self.log_text.append("开始检测...")
    
    def pause_detection(self):
        """暂停/继续检测"""
        if self.video_thread is None:
            return
        
        if self.video_thread.paused:
            self.video_thread.resume()
            self.pause_btn.setText("暂停")
            self.log_text.append("继续检测")
        else:
            self.video_thread.pause()
            self.pause_btn.setText("继续")
            self.log_text.append("暂停检测")
    
    def stop_detection(self):
        """停止检测"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(False)
        
        self.statusBar().showMessage("检测已停止")
        self.log_text.append("检测已停止")
    
    def update_frame(self, frame):
        """更新视频帧"""
        # BGR转RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放以适应标签
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # 更新FPS
        if self.video_thread:
            self.fps_label.setText(f"FPS: {self.video_thread.stats['current_fps']:.1f}")
    
    def update_stats(self, stats):
        """更新统计信息"""
        self.total_label.setText(f"检测总数: {stats['total_count']}")
        self.avg_speed_label.setText(f"平均速度: {stats['avg_speed']:.1f} km/h")
        
        self.car_label.setText(f"轿车: {stats['car_count']}")
        self.truck_label.setText(f"卡车: {stats['truck_count']}")
        self.bus_label.setText(f"公交车: {stats['bus_count']}")
        self.moto_label.setText(f"摩托车: {stats['motorcycle_count']}")
        
        if self.video_thread:
            self.up_count_label.setText(f"上行: {self.video_thread.count_up}")
            self.down_count_label.setText(f"下行: {self.video_thread.count_down}")
    
    def update_detections(self, detections):
        """更新检测日志"""
        if len(detections) > 0:
            # 只在必要时更新日志
            pass
    
    def export_csv(self):
        """导出CSV数据"""
        if self.video_thread is None:
            QMessageBox.warning(self, "警告", "没有检测数据进行导出!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "", "CSV文件 (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Track ID', 'Class', 'Speed (km/h)'])
                    
                    # 获取历史数据
                    for track_id in self.video_thread.speed_estimator.trajectories:
                        class_id = 0  # 需要从跟踪器获取
                        avg_speed = self.video_thread.speed_estimator.get_average_speed(track_id)
                        writer.writerow([track_id, CLASS_NAMES[class_id], f"{avg_speed:.2f}"])
                
                QMessageBox.information(self, "成功", f"数据已导出到: {file_path}")
                self.log_text.append(f"数据已导出: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def save_screenshot(self):
        """保存截图"""
        if self.video_label.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存截图", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)"
        )
        
        if file_path:
            self.video_label.pixmap().save(file_path)
            QMessageBox.information(self, "成功", f"截图已保存: {file_path}")
            self.log_text.append(f"截图已保存: {file_path}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.video_thread is not None:
            self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactoryFusion)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
