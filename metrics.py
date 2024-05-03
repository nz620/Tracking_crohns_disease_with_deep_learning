import torch
import numpy as np
import cv2

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

# class SegMetrics:
#     def __init__(self, metric_names):
#         self.metric_names = metric_names
#         self.results = {name: [] for name in metric_names}
#         self.image_metrics = []

#     def update(self, pred, label, image_name):
#         metrics_result = {}
#         for metric in self.metric_names:
#             if metric == 'iou':
#                 value = iou(pred, label)
#                 self.results['iou'].append(value.tolist()) 
#                 metrics_result['iou'] = value.tolist()
#             elif metric == 'dice':
#                 value = dice(pred, label)
#                 self.results['dice'].append(value.tolist()) 
#                 metrics_result['dice'] = value.tolist()
#         self.image_metrics.append({"image_name": image_name, "metrics": metrics_result})

#     def compute(self):
       
#         final_results = {metric: np.mean(values) for metric, values in self.results.items()}
#         return final_results, self.image_metrics


class SegMetrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.results = {name: [] for name in metric_names}
        self.image_metrics = []

    def update(self, pred, label, image_name):
        metrics_result = {}
        for metric in self.metric_names:
            if metric == 'iou':
                value = iou(pred, label)
                self.results['iou'].append(value.tolist())
                metrics_result['iou'] = value.tolist()
            elif metric == 'dice':
                value = dice(pred, label)
                self.results['dice'].append(value.tolist())
                metrics_result['dice'] = value.tolist()
        self.image_metrics.append({"image_name": image_name, "metrics": metrics_result})

    def compute(self):
        all_results = {}
        outliers_info = {}
        for metric, values in self.results.items():
            values = np.concatenate(values)  
            filtered_values, outlier_indices = self.remove_outliers(values)
            all_results[metric] = {
                'mean_with_outliers': np.mean(values),
                'mean_without_outliers': np.mean(filtered_values),
            }
            outliers_info[metric] = [self.image_metrics[i]['image_name'] for i in outlier_indices]
        
        self.report_outliers(outliers_info)
        return all_results, self.image_metrics

    def remove_outliers(self, data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)
        outliers = np.where(~mask)[0]  # Indices of outliers
        return data[mask], outliers

    def report_outliers(self, outliers_info):
        print("Outliers Detected:")
        for metric, names in outliers_info.items():
            print(f"{metric}: {len(names)} outliers")
            for name in names:
                print(f"  - {name}")
                
