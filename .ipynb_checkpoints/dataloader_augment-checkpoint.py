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
from prompt_generator import generate_bounding_boxes
import re
from skimage import transform, util, filters, exposure
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

def apply_augmentations(img, gt, centreline):
    if random.random() < 0.2:
        # Scaling and rotation
        angle = random.uniform(-180, 180)  # Angle for rotation
        img, gt, centreline = rotate_and_crop_image(img, gt, centreline, angle)

    if random.random() < 0.15:
        # Gaussian noise
        img = util.random_noise(img, mode='gaussian', var=random.uniform(0, 0.1))

    if random.random() < 0.2:
        # Gaussian blur
        sigma = random.uniform(0.5, 1.5)
        img = filters.gaussian(img, sigma=sigma)

    if random.random() < 0.15:
        # Brightness adjustment
        v_factor = random.uniform(0.7, 1.3)
        img = exposure.adjust_gamma(img, gamma=v_factor)

    if random.random() < 0.15:
        # Contrast adjustment
        v_min, v_max = np.percentile(img, (5, 95))  
        img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

    return img, gt, centreline

class NpyDataset(Dataset):
    def __init__(self, data_root, augment=False, aug_num=1,complete_centreline=False):
        self.data_root = data_root
        self.augment = augment
        self.aug_num = aug_num + 1 if augment else 1   # Including the original image
        self.complete_centreline = complete_centreline
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.centreline_path = join(data_root, "centreline")
        self.generated_centreilne_path = join(data_root, "centreline_single")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        if self.complete_centreline:
            self.gt_path_files = [
                file for file in self.gt_path_files
                if  os.path.isfile(join(self.img_path, os.path.basename(file))) and
                    os.path.isfile(join(self.centreline_path, os.path.basename(file))) and
                    os.path.isfile(join(self.generated_centreilne_path, os.path.basename(file)))
            ]
        else:
            self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file))) and
               os.path.isfile(join(self.centreline_path, os.path.basename(file))) and
               self._contains_valid_data(join(self.centreline_path, os.path.basename(file)))
        ]
        print(f"Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files) * self.aug_num
    
    def _contains_valid_data(self, filepath):
        try:
            data = np.load(filepath)
            # Here we check if the file contains any non-zero element
            return np.any(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False

    def __getitem__(self, index):
        original_index = index // self.aug_num
        augmentation_index = index % self.aug_num  # 0 for the original, 1-10 for augmented versions

        img_name = os.path.basename(self.gt_path_files[original_index])
        img_1024 = np.load(join(self.img_path, img_name), allow_pickle=True).copy()
        gt = np.load(join(self.gt_path, img_name), allow_pickle=True).copy()
        centreline = np.load(join(self.centreline_path, img_name), allow_pickle=True).copy()
        if self.complete_centreline:
            generated_centrelines = np.load(join(self.generated_centreilne_path, img_name), allow_pickle=True).copy()
            centreline = np.concatenate([centreline, generated_centrelines], axis=0)
    
        if augmentation_index > 0 and self.augment:
            img_1024, gt, centreline = apply_augmentations(img_1024, gt, centreline)

        gt2D = np.isin(gt, np.unique(gt)[1:]).astype(np.uint8)
        centreline_label = np.ones(centreline.shape[0], dtype=np.int32)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        bbox = generate_bounding_boxes(centreline)
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(centreline).float(),
            torch.tensor(centreline_label).int(),
            torch.tensor(bbox).int(),
            img_name
        )


        
if __name__ == "__main__":
    coronal_dataset_fold0 = NpyDataset('data/centreline_set/coronal/npy2023_5_folds/fold_0',augment=True,aug_num=1,complete_centreline=True)

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
    train_dataloader_fold0 = DataLoader(coronal_dataset_fold0, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
   
   
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    for step, (image, gt, centreline,centreline_label,bbox,names_temp) in enumerate(train_dataloader_fold0):
        # print(image.shape, gt.shape, centreline.shape)
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(25, 25))
        idx = 0
        axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
        show_mask(gt[idx].cpu().numpy(), axs[0])
        # Iterate through each pair of points in the centreline array
        show_points(centreline[idx].cpu().numpy(), centreline_label[0].numpy(), axs[0])
        # for i in range(bbox[idx].shape[0]):
        #     show_box(bbox[idx][i].cpu().numpy(), axs[0])
        
        axs[0].axis("off")
        axs[0].set_title(names_temp[idx])


        show_mask(gt[idx].cpu().numpy(), axs[1])

        # Iterate through each pair of points in the centreline array for the second subplot
        show_points(centreline[idx].cpu().numpy(), centreline_label[0].numpy(), axs[1])
        # for i in range(bbox[idx].shape[0]):
        #     show_box(bbox[idx][i].cpu().numpy(), axs[0])
        axs[1].axis("off")
        axs[1].set_title(names_temp[idx])

        plt.subplots_adjust(wspace=0.01, hspace=0)
        svaed_path = os.path.join("sanitytest_5fold", "data_sanitycheck"+str(names_temp)+str(step)+".png")
        if not os.path.exists("sanitytest_5fold"):
            os.makedirs("sanitytest_5fold")
        plt.savefig(svaed_path, bbox_inches="tight", dpi=100)
        print(f"Saved to {svaed_path}")
        plt.close()
        if step == 20:
            break
    
