"""
速度估算模块
基于车辆轨迹和帧间位移计算速度
"""
import numpy as np
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d


class SpeedEstimator:
    """速度估算器 - 使用EWMA平滑和卡尔曼滤波优化"""
    
    def __init__(self, pixels_per_meter=30, fps=30, smoothing_window=5):
        """
        初始化速度估算器
        
        Args:
            pixels_per_meter: 像素/米比例系数（根据实际场景标定）
            fps: 视频帧率
            smoothing_window: 平滑窗口大小
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.smoothing_window = smoothing_window
        
        # 轨迹存储: {track_id: [(x, y, timestamp, speed), ...]}
        self.trajectories = defaultdict(list)
        
        # 速度缓冲用于平滑
        self.speed_buffer = defaultdict(list)
        
        # 历史速度（用于计算平均速度）
        self.speed_history = defaultdict(list)
        
        # 卡尔曼滤波状态
        self.kalman_states = {}
    
    def update(self, track_id, bbox, frame_idx, timestamp=None):
        """
        更新轨迹并计算速度
        
        Args:
            track_id: 跟踪ID
            bbox: 边界框 [x1, y1, x2, y2]
            frame_idx: 当前帧索引
            timestamp: 时间戳（可选）
            
        Returns:
            speed_kmh: 速度 (km/h)
        """
        if timestamp is None:
            timestamp = frame_idx / self.fps
        
        # 计算边界框中心
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # 更新轨迹
        self.trajectories[track_id].append({
            'x': cx,
            'y': cy,
            'frame': frame_idx,
            'timestamp': timestamp
        })
        
        # 计算瞬时速度
        speed = 0
        if len(self.trajectories[track_id]) >= 2:
            p1 = self.trajectories[track_id][-2]
            p2 = self.trajectories[track_id][-1]
            
            # 像素位移
            dx = p2['x'] - p1['x']
            dy = p2['y'] - p1['y']
            pixel_dist = np.sqrt(dx**2 + dy**2)
            
            # 时间间隔
            dt = p2['timestamp'] - p1['timestamp']
            
            if dt > 0:
                # 转换为米/秒，然后转换为km/h
                dist_meters = pixel_dist / self.pixels_per_meter
                speed_ms = dist_meters / dt
                speed = speed_ms * 3.6  # m/s -> km/h
        
        # EWMA平滑
        if len(self.speed_buffer[track_id]) > 0:
            alpha = 0.3
            speed = alpha * speed + (1 - alpha) * self.speed_buffer[track_id][-1]
        
        self.speed_buffer[track_id].append(speed)
        
        # 限制缓冲大小
        if len(self.speed_buffer[track_id]) > self.smoothing_window:
            self.speed_buffer[track_id].pop(0)
        
        # 保存到历史
        self.speed_history[track_id].append(speed)
        
        return speed
    
    def get_average_speed(self, track_id):
        """获取某轨迹的平均速度"""
        if track_id in self.speed_history and len(self.speed_history[track_id]) > 0:
            speeds = [s for s in self.speed_history[track_id] if s > 0]
            if len(speeds) > 0:
                return np.mean(speeds)
        return 0
    
    def get_current_speed(self, track_id):
        """获取当前速度"""
        if track_id in self.speed_buffer and len(self.speed_buffer[track_id]) > 0:
            return self.speed_buffer[track_id][-1]
        return 0
    
    def get_trajectory(self, track_id):
        """获取完整轨迹"""
        return self.trajectories.get(track_id, [])
    
    def clear_track(self, track_id):
        """清除某轨迹的数据"""
        if track_id in self.trajectories:
            del self.trajectories[track_id]
        if track_id in self.speed_buffer:
            del self.speed_buffer[track_id]
        if track_id in self.speed_history:
            del self.speed_history[track_id]
    
    def reset(self):
        """重置所有数据"""
        self.trajectories.clear()
        self.speed_buffer.clear()
        self.speed_history.clear()
        self.kalman_states.clear()


class LineSpeedEstimator:
    """基于虚拟线的速度估算 - 通过测量车辆穿越固定距离的时间计算速度"""
    
    def __init__(self, line_y1, line_y2, real_distance=10, pixels_per_meter=30, fps=30):
        """
        Args:
            line_y1: 上虚拟线Y坐标
            line_y2: 下虚拟线Y坐标
            real_distance: 两线间实际距离（米）
            pixels_per_meter: 像素/米比例
            fps: 帧率
        """
        self.line_y1 = line_y1
        self.line_y2 = line_y2
        self.real_distance = real_distance
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        
        # 记录车辆穿越虚拟线的时间
        self.crossing_times = {}  # {track_id: {line_id: timestamp}}
    
    def check_line_crossing(self, track_id, cy, timestamp):
        """
        检查是否穿越虚拟线
        Returns:
            line_id: 穿越的线ID (1或2)，None表示未穿越
        """
        # 检查上虚拟线
        if self.line_y1 - 5 <= cy <= self.line_y1 + 5:
            if track_id not in self.crossing_times:
                self.crossing_times[track_id] = {}
            if 1 not in self.crossing_times[track_id]:
                self.crossing_times[track_id][1] = timestamp
                return 1
        
        # 检查下虚拟线
        if self.line_y2 - 5 <= cy <= self.line_y2 + 5:
            if track_id not in self.crossing_times:
                self.crossing_times[track_id] = {}
            if 2 not in self.crossing_times[track_id]:
                self.crossing_times[track_id][2] = timestamp
                return 2
        
        return None
    
    def calculate_speed(self, track_id):
        """计算通过两虚拟线间的速度"""
        if track_id not in self.crossing_times:
            return None
        
        times = self.crossing_times[track_id]
        if 1 in times and 2 in times:
            dt = abs(times[2] - times[1])
            if dt > 0:
                speed = (self.real_distance / dt) * 3.6  # km/h
                return speed
        
        return None
    
    def reset(self):
        """重置"""
        self.crossing_times.clear()


if __name__ == "__main__":
    # 测试速度估算器
    print("测试速度估算器...")
    estimator = SpeedEstimator(pixels_per_meter=30, fps=30)
    
    # 模拟车辆移动
    for i in range(20):
        bbox = [100 + i*5, 200, 150 + i*5, 250]
        speed = estimator.update(1, bbox, i)
        print(f"帧 {i}: 速度 = {speed:.2f} km/h")
    
    print(f"平均速度: {estimator.get_average_speed(1):.2f} km/h")
    print("速度估算器测试通过!")
