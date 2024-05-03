import json
import numpy as np
import matplotlib.pyplot as plt
import os
from loss import FocalDiceloss_IoULoss
from metrics import SegMetrics, _threshold
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
import monai
from sam_med2d import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import re
from utils.display_helper import show_mask, show_points, show_box
import cv2





class Postprocess():
    def __init__(
        self, 
        hole_filling_kernel_size=3,
        holee_filling_iterations=3,
        hole_filling_majority_threshold=1,
        morphological_closing_kernel_size=7):
        
        self.hole_filling_kernel_size = hole_filling_kernel_size
        self.holee_filling_iterations = holee_filling_iterations
        self.hole_filling_majority_threshold = hole_filling_majority_threshold
        self.morphological_closing_kernel_size = morphological_closing_kernel_size
        self.kerne_h = np.ones((self.hole_filling_kernel_size, self.hole_filling_kernel_size), np.uint8)
        self.kernel_m = np.ones((self.morphological_closing_kernel_size, self.morphological_closing_kernel_size), np.uint8)
    def postprocess_mask(self, mask):
        mask = self.fill_holes(mask)
        mask = self.morphological_closing(mask)
        return mask
    
        