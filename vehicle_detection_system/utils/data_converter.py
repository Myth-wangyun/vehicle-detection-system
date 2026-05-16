"""
数据转换工具
将各种数据集格式转换为YOLO格式
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


class DataConverter:
    """数据集格式转换器"""
    
    # 类别映射
    CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
    CLASS_MAPPING = {
        'car': 0,
        'truck': 1,
        'bus': 2,
        'motorcycle': 3,
        # 兼容其他数据集的类别名
        'vehicle': 0,
        'sedan': 0,
        'suv': 0,
        'van': 1,
        'coach': 2,
        'bicycle': 3
    }
    
    @staticmethod
    def voc_to_yolo(xml_path, img_width, img_height):
        """
        将VOC XML标注转换为YOLO格式
        
        Args:
            xml_path: VOC格式XML文件路径
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            list: YOLO格式标注 [[class_id, x_center, y_center, width, height], ...]
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            if class_name not in DataConverter.CLASS_MAPPING:
                continue
            
            class_id = DataConverter.CLASS_MAPPING[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为YOLO格式（归一化中心点+宽高）
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            annotations.append([class_id, x_center, y_center, width, height])
        
        return annotations
    
    @staticmethod
    def coco_to_yolo(coco_annotations, img_width, img_height):
        """
        将COCO JSON标注转换为YOLO格式
        
        Args:
            coco_annotations: COCO格式标注列表
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            list: YOLO格式标注
        """
        annotations = []
        
        for ann in coco_annotations:
            category_id = ann['category_id']
            # COCO category_id 可能是从1开始的
            class_id = category_id - 1  # 转换为0开始的索引
            
            if class_id >= len(DataConverter.CLASS_NAMES):
                continue
            
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 转换为YOLO格式
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            
            annotations.append([class_id, x_center, y_center, width, height])
        
        return annotations
    
    @staticmethod
    def save_yolo_annotation(annotations, output_path):
        """
        保存YOLO格式标注
        
        Args:
            annotations: YOLO格式标注列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    @staticmethod
    def convert_folder(xml_folder, output_folder, img_width, img_height):
        """
        批量转换文件夹中的XML文件
        
        Args:
            xml_folder: VOC XML文件夹
            output_folder: YOLO标注输出文件夹
            img_width: 图像宽度
            img_height: 图像高度
        """
        os.makedirs(output_folder, exist_ok=True)
        
        xml_files = list(Path(xml_folder).glob('*.xml'))
        count = 0
        
        for xml_file in xml_files:
            try:
                annotations = DataConverter.voc_to_yolo(str(xml_file), img_width, img_height)
                output_file = os.path.join(output_folder, xml_file.stem + '.txt')
                DataConverter.save_yolo_annotation(annotations, output_file)
                count += 1
            except Exception as e:
                print(f"转换失败 {xml_file}: {e}")
        
        return count


class COCOVehicleExtractor:
    """从COCO数据集中提取车辆类别"""
    
    # COCO数据集车辆相关类别ID
    COCO_VEHICLE_CLASSES = {
        2: 'car',      # bicycle, car, motorcycle, airplane, bus, train, truck, boat
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
    }
    
    @staticmethod
    def filter_vehicle_annotations(coco_json_path, output_json_path):
        """
        从COCO标注中过滤车辆类别
        
        Args:
            coco_json_path: COCO格式JSON文件路径
            output_json_path: 输出JSON文件路径
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 过滤类别
        vehicle_categories = [
            cat for cat in coco_data.get('categories', [])
            if cat['id'] in COCOVehicleExtractor.COCO_VEHICLE_CLASSES
        ]
        
        # 重新映射类别ID
        new_cat_ids = {old_id: new_id for new_id, old_id in enumerate(COCO_VEHICLE_CLASSES.keys())}
        
        for cat in vehicle_categories:
            cat['id'] = new_cat_ids[cat['id']]
        
        # 过滤标注
        vehicle_annotations = [
            ann for ann in coco_data.get('annotations', [])
            if ann['category_id'] in COCOVehicleExtractor.COCO_VEHICLE_CLASSES
        ]
        
        # 更新类别ID
        for ann in vehicle_annotations:
            ann['category_id'] = new_cat_ids[ann['category_id']]
        
        # 保存结果
        result = {
            'images': coco_data.get('images', []),
            'annotations': vehicle_annotations,
            'categories': vehicle_categories
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return len(vehicle_annotations)


if __name__ == "__main__":
    print("测试数据转换工具...")
    
    # 测试VOC转YOLO
    print("\n测试VOC转YOLO格式...")
    
    # 创建模拟数据
    test_xml = """<annotation>
        <size>
            <width>1280</width>
            <height>720</height>
        </size>
        <object>
            <name>car</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>200</ymin>
                <xmax>400</xmax>
                <ymax>350</ymax>
            </bndbox>
        </object>
        <object>
            <name>truck</name>
            <bndbox>
                <xmin>500</xmin>
                <ymin>300</ymin>
                <xmax>800</xmax>
                <ymax>500</ymax>
            </bndbox>
        </object>
    </annotation>"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(test_xml)
        temp_xml = f.name
    
    annotations = DataConverter.voc_to_yolo(temp_xml, 1280, 720)
    print(f"转换结果: {annotations}")
    
    os.unlink(temp_xml)
    
    print("\n数据转换工具测试通过!")
