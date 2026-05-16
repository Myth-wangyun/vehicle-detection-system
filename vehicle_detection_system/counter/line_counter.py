"""
跨线计数模块
虚拟检测线 + 方向判别 + 防重复计数
"""
import numpy as np
from collections import defaultdict
from datetime import datetime


class LineCounter:
    """跨线计数器"""
    
    def __init__(self, line_start=(0, 360), line_end=(1280, 360), name="default"):
        """
        初始化虚拟检测线
        
        Args:
            line_start: 检测线起点 (x, y)
            line_end: 检测线终点 (x, y)
            name: 检测线名称
        """
        self.name = name
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        
        # 线向量和法向量
        self.line_vec = self.line_end - self.line_start
        self.line_len = np.linalg.norm(self.line_vec)
        self.line_unit = self.line_vec / self.line_len if self.line_len > 0 else self.line_vec
        
        # 法向量（垂直于线）
        self.normal = np.array([-self.line_unit[1], self.line_unit[0]], dtype=np.float32)
        
        # 计数状态
        self.counted_ids = {}  # {track_id: {'direction': int, 'time': timestamp, 'line_side': int}}
        self.count_up = 0      # 上行计数
        self.count_down = 0    # 下行计数
        self.count_left = 0    # 左行计数
        self.count_right = 0   # 右行计数
        
        # 计数历史
        self.count_history = []
        
        # 冷却时间（秒）
        self.cooldown = 2.0
        
        # 上一帧位置
        self.prev_positions = {}  # {track_id: (x, y)}
        
        # 统计信息
        self.stats = {
            'total': 0,
            'by_class': defaultdict(int),
            'avg_speed': 0,
            'speed_distribution': []
        }
    
    def check_cross(self, track_id, bbox, speed=None, timestamp=None):
        """
        检查是否跨线
        
        Args:
            track_id: 跟踪ID
            bbox: 边界框 [x1, y1, x2, y2]
            speed: 当前速度（可选）
            timestamp: 时间戳
            
        Returns:
            dict: {'crossed': bool, 'direction': str, 'track_id': int}
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        # 计算当前中心点
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        curr_pos = np.array([cx, cy], dtype=np.float32)
        
        # 获取上一帧位置
        prev_pos = self.prev_positions.get(track_id)
        
        result = {'crossed': False, 'direction': None, 'track_id': track_id}
        
        # 首次出现，不判断
        if prev_pos is None:
            self.prev_positions[track_id] = curr_pos
            return result
        
        # 检查是否跨线
        # 计算点到直线的有符号距离
        to_point = prev_pos - self.line_start
        from_point = curr_pos - self.line_start
        
        # 上一位置和当前位置在法向量上的投影
        d1 = np.dot(to_point, self.normal)
        d2 = np.dot(from_point, self.normal)
        
        # 判断是否跨线（符号改变）
        if d1 * d2 < 0:  # 跨线
            # 检查冷却时间
            if track_id in self.counted_ids:
                last_time = self.counted_ids[track_id]['time']
                if timestamp - last_time < self.cooldown:
                    self.prev_positions[track_id] = curr_pos
                    return result
            
            # 确定方向
            direction = self._determine_direction(prev_pos, curr_pos)
            
            # 更新计数
            self._update_count(track_id, direction, timestamp, speed)
            
            result['crossed'] = True
            result['direction'] = direction
        
        # 更新上一位置
        self.prev_positions[track_id] = curr_pos
        
        return result
    
    def _determine_direction(self, prev_pos, curr_pos):
        """
        确定穿越方向
        
        Returns:
            str: 'up', 'down', 'left', 'right'
        """
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        
        # 根据线的方向和移动方向确定穿越方向
        # 如果线是水平的（y相近），则上下穿越
        # 如果线是垂直的（x相近），则左右穿越
        
        if abs(self.line_vec[1]) > abs(self.line_vec[0]):  # 近似水平线
            if dy > 0:
                return 'down'
            else:
                return 'up'
        else:  # 近似垂直线
            if dx > 0:
                return 'right'
            else:
                return 'left'
    
    def _update_count(self, track_id, direction, timestamp, speed=None):
        """更新计数"""
        self.counted_ids[track_id] = {
            'direction': direction,
            'time': timestamp,
            'speed': speed
        }
        
        if direction == 'up':
            self.count_up += 1
        elif direction == 'down':
            self.count_down += 1
        elif direction == 'left':
            self.count_left += 1
        elif direction == 'right':
            self.count_right += 1
        
        self.stats['total'] += 1
        
        # 记录历史
        self.count_history.append({
            'track_id': track_id,
            'direction': direction,
            'timestamp': timestamp,
            'speed': speed
        })
        
        # 更新速度统计
        if speed is not None and speed > 0:
            self.stats['speed_distribution'].append(speed)
            if len(self.stats['speed_distribution']) > 0:
                self.stats['avg_speed'] = np.mean(self.stats['speed_distribution'])
    
    def get_count(self):
        """获取当前计数"""
        return {
            'up': self.count_up,
            'down': self.count_down,
            'left': self.count_left,
            'right': self.count_right,
            'total': self.stats['total']
        }
    
    def get_line(self):
        """获取检测线坐标"""
        return (self.line_start.tolist(), self.line_end.tolist())
    
    def reset(self):
        """重置计数"""
        self.counted_ids.clear()
        self.count_up = 0
        self.count_down = 0
        self.count_left = 0
        self.count_right = 0
        self.count_history.clear()
        self.prev_positions.clear()
        self.stats = {
            'total': 0,
            'by_class': defaultdict(int),
            'avg_speed': 0,
            'speed_distribution': []
        }
    
    def is_point_above_line(self, point):
        """判断点是否在线的上方"""
        to_point = np.array(point) - self.line_start
        return np.dot(to_point, self.normal) > 0


class MultiLineCounter:
    """多线计数器（用于多个检测区域）"""
    
    def __init__(self):
        self.counters = {}
        self.next_id = 0
    
    def add_line(self, line_start, line_end, name=None):
        """添加检测线"""
        if name is None:
            name = f"line_{self.next_id}"
            self.next_id += 1
        
        self.counters[name] = LineCounter(line_start, line_end, name)
        return name
    
    def check_cross(self, track_id, bbox, speed=None, timestamp=None):
        """检查所有检测线"""
        results = {}
        for name, counter in self.counters.items():
            result = counter.check_cross(track_id, bbox, speed, timestamp)
            if result['crossed']:
                results[name] = result
        return results
    
    def get_all_counts(self):
        """获取所有计数"""
        counts = {}
        for name, counter in self.counters.items():
            counts[name] = counter.get_count()
        return counts
    
    def reset_all(self):
        """重置所有计数"""
        for counter in self.counters.values():
            counter.reset()


if __name__ == "__main__":
    print("测试跨线计数器...")
    
    # 创建水平检测线
    counter = LineCounter(line_start=(0, 360), line_end=(1280, 360))
    
    # 模拟车辆从上方穿越
    print("\n模拟车辆从上方穿越...")
    for i in range(50):
        bbox = [100, 200 + i*3, 200, 280 + i*3]  # 向下移动
        result = counter.check_cross(1, bbox, speed=60, timestamp=i*0.03)
        if result['crossed']:
            print(f"帧 {i}: 跨线! 方向: {result['direction']}")
    
    print(f"\n最终计数: {counter.get_count()}")
    
    # 模拟车辆从下方穿越
    print("\n模拟车辆从下方穿越...")
    for i in range(50):
        bbox = [300, 500 - i*3, 400, 580 - i*3]  # 向上移动
        result = counter.check_cross(2, bbox, speed=50, timestamp=i*0.03 + 2)
        if result['crossed']:
            print(f"帧 {i}: 跨线! 方向: {result['direction']}")
    
    print(f"\n最终计数: {counter.get_count()}")
    
    print("\n跨线计数器测试通过!")
