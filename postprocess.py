
import numpy as np
import torch
from utils.display_helper import show_mask, show_points, show_box
import SimpleITK as sitk




def morphological_processing_3d(preds, device, kernel_type=sitk.sitkBall, kernel_radius=(7, 7, 7), safe_border=True):
    # Ensure preds is a NumPy array
    if isinstance(preds, torch.Tensor):
        preds_np = preds.cpu().numpy()
    else:
        preds_np = preds

    # Apply binary closing and opening filters
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetSafeBorder(safe_border)
    closing_filter.SetKernelRadius(kernel_radius)
    closing_filter.SetKernelType(kernel_type)

    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetKernelType(kernel_type)
    opening_filter.SetKernelRadius(kernel_radius)

    # Convert 3D numpy array to SimpleITK Image, process it, and convert it back to numpy
    preds_sitk = sitk.GetImageFromArray(preds_np.astype(np.uint8), isVector=False)
    closed_preds = closing_filter.Execute(preds_sitk)
    opened_preds = opening_filter.Execute(closed_preds)
    smoothed_preds_np = sitk.GetArrayFromImage(opened_preds)

    # Convert back to PyTorch tensor
    smoothed_preds = torch.tensor(smoothed_preds_np, dtype=torch.float32, device=device)

    return smoothed_preds
    
        