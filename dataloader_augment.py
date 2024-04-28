import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
import copy
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.display_helper import show_mask, show_points, show_box
import re
from skimage.transform import rotate
import random
random.seed(2023)  
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def flip_image(image, gt, points, flip_horizontal=True, flip_vertical=False):
    # Flip image and ground truth using slicing and ensure positive strides with copy()
    flipped_image = image.copy()
    flipped_gt = gt.copy()

    if flip_horizontal:
        flipped_image = flipped_image[:, ::-1].copy()
        flipped_gt = flipped_gt[:, ::-1].copy()
        points[:, 0] = image.shape[1] - points[:, 0] - 1

    if flip_vertical:
        flipped_image = flipped_image[::-1, :].copy()
        flipped_gt = flipped_gt[::-1, :].copy()
        points[:, 1] = image.shape[0] - points[:, 1] - 1

    return flipped_image, flipped_gt, points


def rotate_and_crop_image(image, gt, points, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Rotate the entire image to fit new dimensions
    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    rotated_gt = cv2.warpAffine(gt, M, (nW, nH))

    # Rotate points using the rotation matrix
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    transformed_points = np.dot(M, points_ones.T).T

    # Find the new center of the rotated image
    new_center_x, new_center_y = nW // 2, nH // 2

    # Crop the rotated image back to original dimensions centered around the new center
    startX = max(new_center_x - w // 2, 0)
    endX = startX + w
    startY = max(new_center_y - h // 2, 0)
    endY = startY + h

    cropped_image = rotated_image[startY:endY, startX:endX]
    cropped_gt = rotated_gt[startY:endY, startX:endX]

    # Adjust points to the new cropped coordinates
    transformed_points[:, 0] -= startX
    transformed_points[:, 1] -= startY

    return cropped_image, cropped_gt, transformed_points



class NpyDataset(Dataset):
    def __init__(self, data_root, augment=False):
        self.data_root = data_root
        self.augment = augment
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.centreline_path = join(data_root, "centreline")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file))) and
               os.path.isfile(join(self.centreline_path, os.path.basename(file)))
        ]
        print(f"Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        # Each image will have 1 original + 5 augmented versions
        if self.augment:
            return len(self.gt_path_files) * 72
        else:
            return len(self.gt_path_files)

    def __getitem__(self, index):
        # Determine the original index and whether it's an augmented version
        original_index = index // 72
        augmentation_type = index % 72  # 0 for original, 1-5 for augmented versions

        img_name = os.path.basename(self.gt_path_files[original_index])
        img_1024 = np.load(join(self.img_path, img_name), allow_pickle=True).copy()
        gt = np.load(join(self.gt_path, img_name), allow_pickle=True).copy()
        centreline = np.load(join(self.centreline_path, img_name), allow_pickle=True).copy()

        if augmentation_type > 0 and self.augment:
            img_1024,gt,centreline = rotate_and_crop_image(img_1024, gt, centreline, 5*augmentation_type)
            
        gt2D = np.isin(gt, np.unique(gt)[1:]).astype(np.uint8)
        centreline_label = np.ones(centreline.shape[0], dtype=np.int32)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(centreline).float(),
            torch.tensor(centreline_label).int(),
            img_name
        )


        
if __name__ == "__main__":
    coronal_dataset = NpyDataset('data/centreline_set/coronal/npy',augment=True)
    # train_dataset = []
    # val_dataset = []
    
    # for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(coronal_dataset)):
    #     match = re.match(r'^(A3|A79|I58)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
    #     if match:
    #         val_dataset.append((image, gt2D, point,point_label, image_name))
    #     else:
    #         train_dataset.append((image, gt2D, point,point_label, image_name))   
    # print('match coronal')     
    # # Split dataset into training and validation
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(coronal_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for step, (image, gt, centreline,centreline_label,names_temp) in enumerate(train_dataloader):
        # print(image.shape, gt.shape, centreline.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        idx = 0
        axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[0])
        # Iterate through each pair of points in the centreline array
        show_points(centreline[idx].cpu().numpy(), centreline_label[0].numpy(), axs[0])
        axs[0].axis("off")
        axs[0].set_title(names_temp[idx])


        show_mask(gt[idx].cpu().numpy(), axs[1])

        # Iterate through each pair of points in the centreline array for the second subplot
        show_points(centreline[idx].cpu().numpy(), centreline_label[0].numpy(), axs[1])
        axs[1].axis("off")
        axs[1].set_title(names_temp[idx])

        plt.subplots_adjust(wspace=0.01, hspace=0)
        svaed_path = os.path.join("sanitytest_augmentation_flip", "data_sanitycheck"+str(names_temp)+str(step)+".png")
        if not os.path.exists("sanitytest_augmentation_flip"):
            os.makedirs("sanitytest_augmentation_flip")
        plt.savefig(svaed_path, bbox_inches="tight", dpi=100)
        print(f"Saved to {svaed_path}")
        plt.close()
        if step == 24:
            break
