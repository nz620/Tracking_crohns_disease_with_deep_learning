
import cv2
import torch
import numpy as np
import torch.nn as nn

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if isinstance(x, list):
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        if x.min() < 0:
            x = m(x)
    return x, y

def get_boundary(mask):
    # Threshold the mask to make it binary
    mask = _threshold(mask, 0.5)
    
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
 
    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze()


    mask_np = mask_np * 255

    if mask_np.shape[0] == 0 or mask_np.shape[1] == 0:
        mask_np = np.zeros((1, 1), dtype=np.uint8)

    if mask_np.ndim != 2:
        raise ValueError("Mask dimensions are not 2D after squeezing. Got dimensions: {}".format(mask_np.shape))

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_np)
    cv2.drawContours(boundary, contours, -1, 1, 1)
    return torch.tensor(boundary, device=mask.device, dtype=torch.float32)

#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=4.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice_loss

class MaskIoULoss(nn.Module):
    def __init__(self):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."
        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class BIoULoss(nn.Module):
    def __init__(self, distance=1):
        super(BIoULoss, self).__init__()
        self.distance = distance

    def forward(self, pred, mask):
        pred_boundary = get_boundary(pred)
        mask_boundary = get_boundary(mask)
        
        pred_boundary_region = self.get_boundary_region(pred_boundary)
        mask_boundary_region = self.get_boundary_region(mask_boundary)
        
        intersection = torch.sum(pred_boundary_region * mask_boundary_region)
        union = torch.sum(pred_boundary_region) + torch.sum(mask_boundary_region) - intersection
        
        return 1 - ((intersection + 1e-7) / (union + 1e-7))

    def get_boundary_region(self, boundary):
        """
        Dilate the boundary to include pixels within a certain distance from the boundary.
        """
        boundary_np = boundary.detach().cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.distance * 2 + 1, self.distance * 2 + 1))
        dilated_boundary = cv2.dilate(boundary_np, kernel)
        
        return torch.tensor(dilated_boundary, device=boundary.device, dtype=torch.float32)

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, mask):
        return self.bce_loss(pred, mask)

class CombinedLoss(nn.Module):
    def __init__(self, weight_focal=1.0, weight_dice=1.0, weight_maskiou=1.0, weight_biou=1.0, weight_bce=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_maskiou = weight_maskiou
        self.weight_biou = weight_biou
        self.weight_bce = weight_bce
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()
        self.biou_loss = BIoULoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, mask, pred_iou):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        # maskiou_loss = self.maskiou_loss(pred, mask, pred_iou)
        biou_loss = self.biou_loss(pred, mask)
        # bce_loss = self.bce_loss(pred.float(), mask.float())

        loss = (
            self.weight_focal * focal_loss +
            self.weight_dice * dice_loss +
            # self.weight_maskiou * maskiou_loss +
            self.weight_biou * biou_loss 
            # self.weight_bce * bce_loss
        )
        return loss