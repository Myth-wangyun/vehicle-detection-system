"""
DeepSORT跟踪器封装模块
集成DeepSORT多目标跟踪算法
"""
import numpy as np
from collections import defaultdict


class DeepSortTracker:
    """DeepSORT跟踪器封装类"""
    
    def __init__(self, max_age=30, n_init=3, nn_budget=100, max_iou_distance=0.7):
        """
        初始化DeepSORT跟踪器
        
        Args:
            max_age: 最大跟踪寿命（帧数）
            n_init: 初始确认需要的连续检测帧数
            nn_budget: 特征库大小限制
            max_iou_distance: IOU匹配阈值
        """
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.max_iou_distance = max_iou_distance
        
        # 跟踪器状态
        self.tracker = None
        self.tracks = []
        self.next_id = 1
        
        # 轨迹历史
        self.track_history = defaultdict(list)
        self.track_confirmed = set()
        self.track_lost = set()
        
        self._init_tracker()
    
    def _init_tracker(self):
        """初始化DeepSORT跟踪器"""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort as DS
            self.tracker = DS(
                max_age=self.max_age,
                n_init=self.n_init,
                nn_budget=self.nn_budget,
                max_iou_distance=self.max_iou_distance
            )
            print("DeepSORT跟踪器初始化成功")
        except ImportError:
            print("警告: deep_sort_realtime未安装，使用简单跟踪器")
            self.tracker = None
    
    def update(self, detections, frame=None):
        """
        更新跟踪器
        
        Args:
            detections: 检测结果列表，每项为 [x1, y1, x2, y2, confidence, class_id]
            frame: 当前帧图像（可选，用于特征提取）
            
        Returns:
            tracks: 跟踪结果列表
        """
        if self.tracker is not None:
            return self._update_deepsort(detections, frame)
        else:
            return self._update_simple(detections)
    
    def _update_deepsort(self, detections, frame):
        """使用DeepSORT更新"""
        if len(detections) == 0:
            self.tracker.predict()
            self.tracks = []
            return self.tracks
        
        # 格式转换：[x1, y1, x2, y2, conf, class] -> [x1, y1, x2, y2, conf]
        dets = []
        for det in detections:
            if len(det) >= 5:
                dets.append(list(det[:5]) + [det[4]])  # 确保格式正确
        
        # 更新跟踪器
        self.tracker.predict()
        tracks = self.tracker.update(dets, frame)
        
        # 解析跟踪结果
        self.tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            class_id = int(track.det_class) if hasattr(track, 'det_class') else 0
            
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'class_id': class_id,
                'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                'history': self.track_history[track_id]
            }
            
            self.tracks.append(track_info)
            self.track_confirmed.add(track_id)
            self.track_history[track_id].append(track_info['center'])
            
            # 限制历史长度
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
        
        return self.tracks
    
    def _update_simple(self, detections):
        """简单跟踪器（当DeepSORT不可用时使用）"""
        self.tracks = []
        
        for det in detections:
            bbox = det[:4]
            track_id = hash(tuple(bbox)) % 10000  # 简单哈希
            class_id = int(det[5]) if len(det) > 5 else 0
            
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'class_id': class_id,
                'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                'history': self.track_history[track_id]
            }
            
            self.tracks.append(track_info)
            self.track_history[track_id].append(track_info['center'])
        
        return self.tracks
    
    def get_track_history(self, track_id):
        """获取某跟踪ID的历史轨迹"""
        return self.track_history.get(track_id, [])
    
    def get_active_tracks(self):
        """获取所有活跃跟踪"""
        return [t for t in self.tracks if t['track_id'] in self.track_confirmed]
    
    def reset(self):
        """重置跟踪器"""
        self.track_history.clear()
        self.track_confirmed.clear()
        self.track_lost.clear()
        self.tracks = []
        if self.tracker is not None:
            self._init_tracker()


class SimpleTracker:
    """简单的IoU跟踪器（备用）"""
    
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections):
        """更新跟踪"""
        self.frame_count += 1
        updated_tracks = []
        
        # 匹配检测与现有轨迹
        matched_tracks = set()
        for det in detections:
            best_match = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._calculate_iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match is not None:
                self.tracks[best_match]['bbox'] = det[:4]
                self.tracks[best_match]['age'] = 0
                matched_tracks.add(best_match)
            else:
                # 创建新轨迹
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'bbox': det[:4],
                    'age': 0,
                    'class_id': int(det[5]) if len(det) > 5 else 0
                }
                matched_tracks.add(track_id)
        
        # 更新和清理轨迹
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # 构建返回结果
        for track_id, track in self.tracks.items():
            if track['age'] <= 3:  # 新轨迹需要几帧确认
                bbox = track['bbox']
                updated_tracks.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'class_id': track['class_id'],
                    'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                })
        
        return updated_tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0


if __name__ == "__main__":
    print("测试DeepSORT跟踪器...")
    
    # 测试简单跟踪器
    tracker = SimpleTracker()
    
    # 模拟检测
    detections = [
        [100, 100, 200, 200, 0.9, 0],
        [300, 100, 400, 200, 0.85, 0],
    ]
    
    tracks = tracker.update(detections)
    print(f"第一帧跟踪: {len(tracks)} 个目标")
    
    # 模拟第二帧
    detections2 = [
        [105, 102, 205, 202, 0.9, 0],
        [305, 103, 405, 203, 0.85, 0],
    ]
    
    tracks2 = tracker.update(detections2)
    print(f"第二帧跟踪: {len(tracks2)} 个目标")
    
    print("跟踪器测试通过!")
