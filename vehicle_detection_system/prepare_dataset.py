"""
数据集准备脚本
支持 UA-DETRAC、KITTI、Bdd100K 等数据集转换为 YOLO 格式
"""
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import random

# UA-DETRAC 数据集下载链接
UA_DETRAC_URL = "https://detrac-db.rit.albany.edu/widget/dataset-details/UA-DETRAC"
BACKUP_URLS = [
    # 备用下载源（如果主站不可用）
]

class DatasetPreparer:
    """数据集准备工具"""
    
    # 类别映射：UA-DETRAC -> YOLO
    CLASS_MAPPING = {
        'car': 0,
        'van': 1,
        'bus': 2,
        'other': 1,  # 将其他类型映射到 truck
    }
    
    def __init__(self, dataset_root="./data/vehicle_dataset"):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / "images"
        self.labels_dir = self.dataset_root / "labels"
        
    def create_dirs(self):
        """创建数据集目录结构"""
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        print(f"数据集目录已创建: {self.dataset_root}")
        
    def download_detrac(self, dest_path="./UA-DETRAC.zip"):
        """
        下载 UA-DETRAC 数据集

        注意：由于 UA-DETRAC 需要注册，这里提供手动下载说明
        """
        print("=" * 60)
        print("UA-DETRAC 数据集下载说明")
        print("=" * 60)
        print("""
由于 UA-DETRAC 数据集需要官方注册才能下载，
请按以下步骤操作：

1. 访问: https://detrac-db.rit.albany.edu/Data
2. 注册账号并申请数据集访问权限
3. 下载训练集和测试集 ZIP 文件
4. 将文件放置在项目根目录

或者使用其他公开数据集：
- BDD100K: https://bdd-data.berkeley.edu/
- KITTI: http://www.cvlibs.net/datasets/kitti/
""")
        return False
        
    def convert_detrac_to_yolo(self, xml_dir, img_dir, output_dir, split='train'):
        """
        将 UA-DETRAC XML 标注转换为 YOLO 格式

        Args:
            xml_dir: XML 标注文件目录
            img_dir: 图像文件目录
            output_dir: 输出目录
            split: 数据集划分 (train/val/test)
        """
        xml_dir = Path(xml_dir)
        output_img_dir = self.images_dir / split
        output_label_dir = self.labels_dir / split
        
        # 处理每个序列的 XML
        for xml_file in xml_dir.glob("*.xml"):
            # 解析 XML 获取图像列表和标注
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 获取序列名
            sequence_name = xml_file.stem
            
            for frame in root.findall('.//frame'):
                frame_id = frame.get('num')
                
                # 获取图像文件名
                img_name = f"{sequence_name}_img{frame_id.zfill(5)}.jpg"
                src_img = Path(img_dir) / sequence_name / img_name
                
                if not src_img.exists():
                    continue
                    
                # 创建输出文件名
                output_img_path = output_img_dir / f"{sequence_name}_{frame_id.zfill(5)}.jpg"
                output_label_path = output_label_dir / f"{sequence_name}_{frame_id.zfill(5)}.txt"
                
                # 复制图像
                shutil.copy(src_img, output_img_path)
                
                # 转换标注
                self._convert_frame_annotations(frame, output_label_path)
                
        print(f"已转换 {split} 集")
        
    def _convert_frame_annotations(self, frame_elem, output_path):
        """转换单帧标注"""
        img_width = 960  # UA-DETRAC 默认宽度
        img_height = 540  # UA-DETRAC 默认高度
        
        with open(output_path, 'w') as f:
            for target in frame_elem.findall('.//target'):
                vehicle_type = target.get('type', '').lower()
                
                if vehicle_type not in self.CLASS_MAPPING:
                    continue
                    
                class_id = self.CLASS_MAPPING[vehicle_type]
                
                # 获取边界框
                box = target.find('box')
                if box is None:
                    continue
                    
                left = float(box.find('left').text)
                top = float(box.find('top').text)
                width = float(box.find('width').text)
                height = float(box.find('height').text)
                
                # 转换为 YOLO 格式
                x_center = (left + width / 2) / img_width
                y_center = (top + height / 2) / img_height
                w_norm = width / img_width
                h_norm = height / img_height
                
                # 限制范围
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0.001, min(1, w_norm))
                h_norm = max(0.001, min(1, h_norm))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
    def create_sample_dataset(self, num_samples=100):
        """
        创建示例数据集（用于快速测试）
        从公开资源生成测试数据
        """
        print("创建示例数据集...")
        
        import cv2
        import numpy as np
        
        # 创建合成的车辆图像
        for i in range(num_samples):
            # 生成随机背景
            bg = np.random.randint(40, 100, (540, 960, 3), dtype=np.uint8)
            
            # 添加一些"车辆"
            num_cars = np.random.randint(2, 8)
            for _ in range(num_cars):
                x = np.random.randint(50, 850)
                y = np.random.randint(50, 450)
                w = np.random.randint(60, 150)
                h = np.random.randint(40, 100)
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                cv2.rectangle(bg, (x, y), (x+w, y+h), color, -1)
            
            # 保存图像
            split = 'train' if i < num_samples * 0.7 else 'val'
            img_path = self.images_dir / split / f"sample_{i:04d}.jpg"
            cv2.imwrite(str(img_path), bg)
            
            # 保存标注
            label_path = self.labels_dir / split / f"sample_{i:04d}.txt"
            with open(label_path, 'w') as f:
                for _ in range(num_cars):
                    # 假设图像中随机有 2-7 辆车
                    pass  # 示例数据不生成真实标注
                    
        print(f"已创建 {num_samples} 个示例图像")
        
    def prepare_dataset_config(self):
        """生成 YOLO 数据集配置文件"""
        config_path = self.dataset_root / "dataset.yaml"
        
        config_content = f"""# 车辆检测数据集配置
path: {self.dataset_root.absolute()}
train: images/train
val: images/val
test: images/test

# 类别
names:
  0: car
  1: truck
  2: bus

nc: 3
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        print(f"数据集配置已保存: {config_path}")
        return config_path


class KITTIPreparer:
    """KITTI 数据集转换为 YOLO 格式"""
    
    # KITTI 类别映射到 YOLO 类别
    # KITTI 类别: Car, Pedestrian, Cyclist, Van, Truck, Misc, Tram, Person_sitting, Don'tCare
    CLASS_MAPPING = {
        'Car': 0,        # 汽车
        'Van': 1,        # 货车/面包车
        'Truck': 1,      # 卡车 -> 归类为货车
        'Pedestrian': 2, # 行人
        'Cyclist': 3,    # 骑行者
        'Tram': 4,       # 有轨电车
    }
    
    # 用于车辆检测的类别（你论文需要的）
    VEHICLE_CLASSES = {'Car', 'Van', 'Truck', 'Tram'}
    
    def __init__(self, kitti_root, output_root):
        """
        Args:
            kitti_root: KITTI 数据集根目录 (包含 training/ 和 testing/)
            output_root: YOLO 格式输出目录
        """
        self.kitti_root = Path(kitti_root)
        self.output_root = Path(output_root)
        self.train_images = self.output_root / 'images' / 'train'
        self.train_labels = self.output_root / 'labels' / 'train'
        self.val_images = self.output_root / 'images' / 'val'
        self.val_labels = self.output_root / 'labels' / 'val'
        
    def create_dirs(self):
        """创建输出目录"""
        for d in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"输出目录已创建: {self.output_root}")
    
    def parse_kitti_label(self, label_path):
        """
        解析 KITTI 标注文件
        KITTI 格式: class truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom 
                   3d_height 3d_width 3d_length 3d_x 3d_y 3d_z rotation_y score
        """
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                cls = parts[0]
                
                # 跳过 Don'tCare 和非车辆类别（除非你需要行人/骑行者）
                if cls == 'DontCare' or cls not in self.CLASS_MAPPING:
                    continue
                
                # 只保留车辆类别
                if cls not in self.VEHICLE_CLASSES:
                    continue
                
                # 解析边界框
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                annotations.append({
                    'class': cls,
                    'class_id': self.CLASS_MAPPING[cls],
                    'bbox': [left, top, right, bottom]
                })
        
        return annotations
    
    def convert_kitti_to_yolo(self, img_path, annotations, img_width, img_height):
        """
        将 KITTI 边界框转换为 YOLO 格式
        YOLO 格式: class_id center_x center_y width height (归一化到 0-1)
        """
        yolo_annotations = []
        
        for ann in annotations:
            left, top, right, bottom = ann['bbox']
            
            # 计算中心点和宽高
            x_center = (left + right) / 2.0
            y_center = (top + bottom) / 2.0
            width = right - left
            height = bottom - top
            
            # 归一化
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            # 确保值在 0-1 范围内
            x_center_norm = max(0, min(1, x_center_norm))
            y_center_norm = max(0, min(1, y_center_norm))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            yolo_annotations.append(
                f"{ann['class_id']} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            )
        
        return yolo_annotations
    
    def convert(self, val_split=0.2, copy_images=True):
        """
        执行转换
        
        Args:
            val_split: 验证集比例 (默认 20%)
            copy_images: 是否复制图像 (False 则创建软链接)
        """
        print("=" * 60)
        print("开始转换 KITTI 数据集")
        print("=" * 60)
        
        self.create_dirs()
        
        label_dir = self.kitti_root / 'training' / 'label_2'
        image_dir = self.kitti_root / 'training' / 'image_2'
        
        if not label_dir.exists():
            print(f"错误: 标注目录不存在: {label_dir}")
            print("请确保已下载并解压 KITTI 标注文件")
            return False
        
        if not image_dir.exists():
            print(f"错误: 图像目录不存在: {image_dir}")
            print("请确保已下载并解压 KITTI 图像文件")
            return False
        
        # 获取所有标注文件
        label_files = sorted(list(label_dir.glob('*.txt')))
        print(f"找到 {len(label_files)} 个标注文件")
        
        # 随机打乱并划分
        random.seed(42)  # 可复现
        random.shuffle(label_files)
        
        val_count = int(len(label_files) * val_split)
        train_files = label_files[val_count:]
        val_files = label_files[:val_count]
        
        print(f"训练集: {len(train_files)} 张")
        print(f"验证集: {len(val_files)} 张")
        
        # 转换训练集
        converted_train = 0
        for label_file in train_files:
            img_name = label_file.stem + '.png'  # KITTI 使用 PNG
            img_path = image_dir / img_name
            
            if not img_path.exists():
                img_path = image_dir / (label_file.stem + '.jpg')  # 备选 jpg
            
            if not img_path.exists():
                continue
            
            # 解析标注
            annotations = self.parse_kitti_label(label_file)
            if not annotations:
                continue
            
            # 获取图像尺寸（KITTI 图像尺寸固定为 1242 x 375）
            img_width, img_height = 1242, 375
            
            # 转换为 YOLO 格式
            yolo_annotations = self.convert_kitti_to_yolo(img_path, annotations, img_width, img_height)
            
            # 复制/链接图像
            if copy_images:
                dst_img = self.train_images / img_name
                shutil.copy2(img_path, dst_img)
            else:
                dst_img = self.train_images / img_name
                if not dst_img.exists():
                    os.symlink(img_path.absolute(), dst_img)
            
            # 保存标注
            dst_label = self.train_labels / f"{img_name.replace('.png', '.txt').replace('.jpg', '.txt')}"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_train += 1
        
        print(f"转换训练集: {converted_train} 张")
        
        # 转换验证集
        converted_val = 0
        for label_file in val_files:
            img_name = label_file.stem + '.png'
            img_path = image_dir / img_name
            
            if not img_path.exists():
                img_path = image_dir / (label_file.stem + '.jpg')
            
            if not img_path.exists():
                continue
            
            annotations = self.parse_kitti_label(label_file)
            if not annotations:
                continue
            
            img_width, img_height = 1242, 375
            yolo_annotations = self.convert_kitti_to_yolo(img_path, annotations, img_width, img_height)
            
            if copy_images:
                dst_img = self.val_images / img_name
                shutil.copy2(img_path, dst_img)
            else:
                dst_img = self.val_images / img_name
                if not dst_img.exists():
                    os.symlink(img_path.absolute(), dst_img)
            
            dst_label = self.val_labels / f"{img_name.replace('.png', '.txt').replace('.jpg', '.txt')}"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_val += 1
        
        print(f"转换验证集: {converted_val} 张")
        
        # 生成数据集配置文件
        self.create_dataset_yaml()
        
        print("=" * 60)
        print("转换完成!")
        print("=" * 60)
        
        return True
    
    def create_dataset_yaml(self):
        """创建 YOLO 数据集配置文件"""
        config_content = f"""# KITTI 车辆检测数据集 (YOLO 格式)
# 转换自: {self.kitti_root}

path: {self.output_root.absolute()}
train: images/train
val: images/val

# 类别 (4 类)
names:
  0: car
  1: truck
  2: pedestrian
  3: cyclist
  4: tram

nc: 5
"""
        config_path = self.output_root / 'kitti.yaml'
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"数据集配置已保存: {config_path}")
        return config_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument('--kitti', type=str, help='KITTI 数据集根目录路径')
    parser.add_argument('--convert', type=str, help='转换现有数据集路径 (VOC/COCO)')
    parser.add_argument('--output', type=str, default='./datasets/kitti_yolo', help='输出目录')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例 (默认 0.2)')
    parser.add_argument('--sample', action='store_true', help='创建示例数据集')
    args = parser.parse_args()
    
    if args.kitti:
        # KITTI 转换
        preparer = KITTIPreparer(args.kitti, args.output)
        preparer.convert(val_split=args.val_split)
    elif args.convert:
        preparer = DatasetPreparer(args.output)
        preparer.convert_detrac_to_yolo(args.convert)
    elif args.sample:
        preparer = DatasetPreparer(args.output)
        preparer.create_dirs()
        preparer.create_sample_dataset()
        preparer.prepare_dataset_config()
    else:
        print("=" * 60)
        print("数据集准备工具")
        print("=" * 60)
        print("\n用法:")
        print("  # 转换 KITTI 数据集:")
        print("  python prepare_dataset.py --kitti /path/to/KITTI --output ./datasets/kitti_yolo")
        print("")
        print("  # 转换 VOC 格式数据集:")
        print("  python prepare_dataset.py --convert /path/to/VOC --output ./datasets/voc_yolo")
        print("")
        print("  # 创建示例数据集:")
        print("  python prepare_dataset.py --sample")
        print("")
        print("=" * 60)
        print("\nKITTI 数据集下载:")
        print("  http://www.cvlibs.net/datasets/kitti/evaluation_object.php")
        print("  需要下载:")
        print("    - Training labels (标注文件)")
        print("    - Training images (图像文件)")
        print("=" * 60)
        

if __name__ == "__main__":
    main()
