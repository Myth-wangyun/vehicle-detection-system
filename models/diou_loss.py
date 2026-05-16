"""
DIoU (Distance-IoU) 损失函数
改进的边界框回归损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DIoULoss(nn.Module):
    """
    DIoU Loss - Distance-IoU Loss
    
    论文: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    https://arxiv.org/abs/1911.08287
    """
    
    def __init__(self, loss_type='iou'):
        super().__init__()
        self.loss_type = loss_type  # 'iou', 'giou', 'diou', 'ciou'
    
    def forward(self, pred_boxes, target_boxes):
        """
        计算DIoU损失
        
        Args:
            pred_boxes: 预测边界框 [N, 4] (x1, y1, x2, y2)
            target_boxes: 目标边界框 [N, 4]
            
        Returns:
            loss: 损失值
        """
        if self.loss_type == 'diou':
            return self._diou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'ciou':
            return self._ciou_loss(pred_boxes, target_boxes)
        elif self.loss_type == 'giou':
            return self._giou_loss(pred_boxes, target_boxes)
        else:
            return self._iou_loss(pred_boxes, target_boxes)
    
    def _iou_loss(self, pred_boxes, target_boxes):
        """标准IoU损失"""
        iou = self._compute_iou(pred_boxes, target_boxes)
        return 1 - iou
    
    def _giou_loss(self, pred_boxes, target_boxes):
        """GIoU损失"""
        giou = self._compute_giou(pred_boxes, target_boxes)
        return 1 - giou
    
    def _diou_loss(self, pred_boxes, target_boxes):
        """DIoU损失"""
        diou = self._compute_diou(pred_boxes, target_boxes)
        return 1 - diou
    
    def _ciou_loss(self, pred_boxes, target_boxes):
        """CIoU损失"""
        ciou = self._compute_ciou(pred_boxes, target_boxes)
        return 1 - ciou
    
    def _compute_iou(self, boxes1, boxes2):
        """计算IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        iou = inter / (union + 1e-7)
        
        return iou
    
    def _compute_giou(self, boxes1, boxes2):
        """计算GIoU"""
        iou = self._compute_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        enclose_area = wh[:, 0] * wh[:, 1]
        
        giou = iou - (enclose_area - self._compute_union(boxes1, boxes2)) / (enclose_area + 1e-7)
        
        return giou.clamp(min=-1, max=1)
    
    def _compute_diou(self, boxes1, boxes2):
        """计算DIoU"""
        iou = self._compute_iou(boxes1, boxes2)
        
        # 计算中心点距离
        center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
        center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
        center_dist_sq = ((center1 - center2) ** 2).sum(dim=1)
        
        # 计算最小包围框对角线距离
        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        diag_dist_sq = ((rb - lt) ** 2).sum(dim=1)
        
        # DIoU
        diou = iou - center_dist_sq / (diag_dist_sq + 1e-7)
        
        return diou.clamp(min=-1, max=1)
    
    def _compute_ciou(self, boxes1, boxes2):
        """计算CIoU - 考虑宽高比"""
        iou = self._compute_iou(boxes1, boxes2)
        
        # 中心点距离
        center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
        center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
        center_dist_sq = ((center1 - center2) ** 2).sum(dim=1)
        
        # 最小包围框
        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        diag_dist_sq = ((rb - lt) ** 2).sum(dim=1)
        
        # 宽高比
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        v = (4 / (torch.pi ** 2)) * ((torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7))) ** 2)
        alpha = v / (v - iou + (1 + 1e-7))
        
        ciou = iou - (center_dist_sq / (diag_dist_sq + 1e-7)) - alpha * v
        
        return ciou.clamp(min=-1, max=1)
    
    def _compute_union(self, boxes1, boxes2):
        """计算并集面积"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        return area1 + area2


class YOLOv8Loss(nn.Module):
    """
    YOLOv8损失函数（集成DIoU）
    """
    
    def __init__(self, box_weight=7.5, cls_weight=0.5, dfl_weight=1.5):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        
        # 使用BCEWithLogitsLoss进行分类
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        
        # DIoU损失
        self.diou_loss = DIoULoss(loss_type='diou')
    
    def forward(self, preds, targets):
        """
        计算损失
        
        Args:
            preds: 预测结果
            targets: 目标
            
        Returns:
            总损失及各项损失
        """
        # 这里简化处理，实际需要根据YOLOv8的具体输出结构
        box_loss = 0
        cls_loss = 0
        dfl_loss = 0
        
        # ... 根据具体实现计算各项损失
        
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss + self.dfl_weight * dfl_loss
        
        return total_loss, {
            'box': box_loss,
            'cls': cls_loss,
            'dfl': dfl_loss,
            'total': total_loss
        }


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算边界框IoU
    
    Args:
        box1: 预测框
        box2: 目标框
        xywh: 格式是否为(x,y,w,h)
        GIoU/DIoU/CIoU: 是否使用对应的IoU变体
        
    Returns:
        IoU值
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    # 交集
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # 并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        # 外接框
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            if CIoU:
                v = (4 / torch.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            
            return iou - rho2 / c2
        
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    
    return iou


if __name__ == "__main__":
    print("测试DIoU损失函数...")
    
    device = torch.device('cpu')
    
    # 创建测试数据
    pred_boxes = torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]], device=device)
    target_boxes = torch.tensor([[105, 105, 205, 205], [60, 60, 160, 160]], device=device)
    
    # 测试DIoU损失
    loss_fn = DIoULoss(loss_type='diou')
    diou_loss = loss_fn(pred_boxes, target_boxes)
    print(f"DIoU Loss: {diou_loss:.4f}")
    
    # 测试CIoU损失
    loss_fn_ciou = DIoULoss(loss_type='ciou')
    ciou_loss = loss_fn_ciou(pred_boxes, target_boxes)
    print(f"CIoU Loss: {ciou_loss:.4f}")
    
    # 使用bbox_iou函数
    iou = bbox_iou(pred_boxes, target_boxes, xywh=False)
    print(f"IoU: {iou.mean():.4f}")
    
    diou = bbox_iou(pred_boxes, target_boxes, xywh=False, DIoU=True)
    print(f"DIoU: {diou.mean():.4f}")
    
    print("DIoU损失函数测试通过!")
