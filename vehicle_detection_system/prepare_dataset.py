"""
数据集准备脚本
下载并转换 UA-DETRAC 数据集为 YOLO 格式
"""
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument('--download', action='store_true', help='下载 UA-DETRAC 数据集')
    parser.add_argument('--convert', type=str, help='转换现有数据集路径')
    parser.add_argument('--sample', action='store_true', help='创建示例数据集')
    parser.add_argument('--output', type=str, default='./data/vehicle_dataset', help='输出目录')
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.output)
    
    if args.download:
        preparer.download_detrac()
    elif args.convert:
        preparer.convert_detrac_to_yolo(args.convert)
    elif args.sample:
        preparer.create_dirs()
        preparer.create_sample_dataset()
        preparer.prepare_dataset_config()
    else:
        preparer.create_dirs()
        preparer.prepare_dataset_config()
        print("\n数据集目录结构已创建。请手动下载 UA-DETRAC 数据集并放置。")
        print("数据集配置已准备好。")
        

if __name__ == "__main__":
    main()
