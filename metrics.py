import torch
import numpy as np

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(torch.float32)  
    return x

def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    if x.min() < 0 or x.max() > 1:
        x = m(x)
    return x, y

def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = _list_tensor(pr, gt)
    pr = _threshold(pr, threshold=threshold)
    gt = _threshold(gt, threshold=threshold)
    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection
    return ((intersection + eps) / (union + eps)).item()  

def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = _list_tensor(pr, gt)
    pr = _threshold(pr, threshold=threshold)
    gt = _threshold(gt, threshold=threshold)
    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr)
    return ((2 * intersection + eps) / (union + eps)).item()  

class SegMetrics:
    def __init__(self, metric_names, device):
        self.metric_names = metric_names
        self.results = {name: [] for name in metric_names}
        self.image_metrics = []
        self.device = device

    def update(self, pred, label, image_name):
        metrics_result = {}
        for metric in self.metric_names:
            if metric == 'iou':
                value = iou(pred, label)
                self.results['iou'].append(value)
                metrics_result['iou'] = value
            elif metric == 'dice':
                value = dice(pred, label)
                self.results['dice'].append(value)
                metrics_result['dice'] = value

        self.image_metrics.append({"image_name": image_name, "metrics": metrics_result})

    def compute(self):
        all_results = {}
        for metric, values in self.results.items():
            values_array = np.array(values)
            mean_value = np.mean(values_array)
            max_value = np.max(values_array)
            min_value = np.min(values_array)
            std = np.std(values_array)
            all_results[metric] = {
                'mean': mean_value,
                'standard deviation': std,
                'max': max_value,
                'min': min_value
            }
        return all_results, self.image_metrics


