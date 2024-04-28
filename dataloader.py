import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.display_helper import show_mask, show_points, show_box
import re

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6




class NpyDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.centreline_path = join(data_root, "centreline")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.centreline_path, os.path.basename(file)))
        ]
        
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        ##WHY THIS LINE?
        gt2D = np.uint8(
            gt == label_ids)
          # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        
        # load centreline 
        centreline_name = os.path.basename(self.gt_path_files[index])
        centreline = np.load(
            join(self.centreline_path, centreline_name), "r", allow_pickle=True
        ) 
        centreline_label = np.ones(centreline.shape[0])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(centreline).float(),
            torch.tensor(centreline_label).int(),
            img_name,
        )
if __name__ == "__main__":
    axial_dataset = NpyDataset('data/centreline_set/axial/npy')
    coronal_dataset = NpyDataset('data/centreline_set/coronal/npy')
    train_dataset = []
    val_dataset = []
    for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(axial_dataset)):
        match = re.match(r'^(A5|A97|I59)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
        if match:
            val_dataset.append((image, gt2D, point,point_label, image_name))
        else:
            train_dataset.append((image, gt2D, point,point_label, image_name))    
    print('match axial')
    for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(coronal_dataset)):
        match = re.match(r'^(A3|A79|I58)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
        if match:
            val_dataset.append((image, gt2D, point,point_label, image_name))
        else:
            train_dataset.append((image, gt2D, point,point_label, image_name))   
    print('match coronal')     
    # # Split dataset into training and validation
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    for step, (image, gt, centreline,centreline_label,names_temp) in enumerate(train_dataloader):
        # print(image.shape, gt.shape, centreline.shape)
        # show the example
        if  step%50 == 0:
            _, axs = plt.subplots(1, 2, figsize=(25, 25))
            idx = 0
            axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
            show_mask(gt[idx].cpu().numpy(), axs[0])
            points_label =  np.ones(centreline[idx].shape[0])
            # Iterate through each pair of points in the centreline array
            show_points(centreline[idx].cpu().numpy(), points_label, axs[0])
            axs[0].axis("off")
            axs[0].set_title(names_temp[idx])


            show_mask(gt[idx].cpu().numpy(), axs[1])

            # Iterate through each pair of points in the centreline array for the second subplot
            show_points(centreline[idx].cpu().numpy(), np.ones(centreline[idx].shape[0]), axs[1])
            axs[1].axis("off")
            axs[1].set_title(names_temp[idx])

            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig("./sanitytest_multimodal/data_sanitycheck"+str(names_temp)+str(step)+".png", bbox_inches="tight", dpi=300)
            plt.close()
    