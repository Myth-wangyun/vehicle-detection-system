"""
数据增强工具
Mosaic + MixUp + 其他增强
"""
import random
import numpy as np
import cv2
from pathlib import Path


class MosaicAugmentation:
    """Mosaic数据增强 - 将4张图像拼接成一张"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def __call__(self, images, labels):
        """
        执行Mosaic增强
        
        Args:
            images: 4张图像列表
            labels: 4组标注列表
            
        Returns:
            mosaic_img: 拼接后的图像
            mosaic_labels: 拼接后的标注
        """
        assert len(images) == 4, "Mosaic需要4张图像"
        
        # 创建画布
        mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []
        
        # 定义4张图像的位置
        yc, xc = [int(random.uniform(self.img_size // 2, self.img_size)) for _ in range(2)]
        
        for i, (img, label) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            
            # 确定位置
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # 右上
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # 左下
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # 右下
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            
            # 粘贴图像
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # 调整标注框
            padw = x1a - x1b
            padh = y1a - y1b
            
            if label is not None and len(label) > 0:
                labels_copy = label.copy()
                labels_copy[:, 1:] = transform_boxes(labels_copy[:, 1:], (w, h), (x1a, y1a), self.img_size * 2)
                mosaic_labels.append(labels_copy)
        
        # 合并所有标注
        if len(mosaic_labels) > 0:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
        else:
            mosaic_labels = np.zeros((0, 5))
        
        # 调整图像大小
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        return mosaic_img, mosaic_labels


class MixUpAugmentation:
    """MixUp数据增强"""
    
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, img1, labels1, img2, labels2):
        """
        执行MixUp增强
        
        Args:
            img1: 图像1
            labels1: 标注1
            img2: 图像2
            labels2: 标注2
            
        Returns:
            mixed_img: 混合后的图像
            mixed_labels: 混合后的标注
        """
        # 随机混合比例
        r = np.random.beta(self.alpha, self.beta)
        
        # 混合图像
        mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        
        # 合并标注
        mixed_labels = np.concatenate([labels1, labels2], 0)
        
        return mixed_img, mixed_labels


class RandomAugmentation:
    """随机数据增强"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, labels):
        """
        应用随机增强
        
        Args:
            img: 输入图像
            labels: 标注
            
        Returns:
            img: 增强后的图像
            labels: 标注
        """
        h, w = img.shape[:2]
        
        # 随机翻转
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]  # 翻转x中心
        
        # 随机亮度
        if random.random() < self.p:
            factor = random.uniform(0.5, 1.5)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # 随机对比度
        if random.random() < self.p:
            factor = random.uniform(0.5, 1.5)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # 随机饱和度
        if random.random() < self.p:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        # 随机模糊
        if random.random() < self.p * 0.3:
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        return img, labels


def transform_boxes(boxes, orig_size, new_origin, new_size):
    """
    变换标注框坐标
    
    Args:
        boxes: 标注框 [N, 5] (class, x, y, w, h)
        orig_size: 原始图像尺寸 (w, h)
        new_origin: 新原点 (x, y)
        new_size: 新尺寸
        
    Returns:
        变换后的标注框
    """
    boxes_transformed = boxes.copy()
    
    # 平移
    boxes_transformed[:, 1] -= new_origin[0]
    boxes_transformed[:, 2] -= new_origin[1]
    
    # 缩放
    scale = new_size / orig_size[0]
    boxes_transformed[:, 1:] *= scale
    
    return boxes_transformed


class AugmentationPipeline:
    """数据增强流水线"""
    
    def __init__(self, mosaic=True, mixup=True, random_aug=True, p=0.5):
        self.mosaic = MosaicAugmentation() if mosaic else None
        self.mixup = MixUpAugmentation() if mixup else None
        self.random_aug = RandomAugmentation(p) if random_aug else None
    
    def __call__(self, images, labels):
        """
        应用增强流水线
        
        Args:
            images: 图像列表
            labels: 标注列表
            
        Returns:
            增强后的图像和标注
        """
        # Mosaic
        if self.mosaic is not None and len(images) == 4:
            images, labels = self.mosaic(images, labels)
        else:
            images = images[0] if len(images) == 1 else images
            labels = labels[0] if len(labels) == 1 else labels
        
        # Random Augmentation
        if self.random_aug is not None:
            images, labels = self.random_aug(images, labels)
        
        return images, labels


if __name__ == "__main__":
    print("测试数据增强工具...")
    
    # 创建模拟图像
    img1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    labels1 = np.array([[0, 0.3, 0.4, 0.2, 0.3]])
    
    # 测试随机增强
    aug = RandomAugmentation()
    img_aug, labels_aug = aug(img1, labels1)
    print(f"随机增强: 输入形状 {img1.shape}, 输出形状 {img_aug.shape}")
    
    print("数据增强工具测试通过!")
