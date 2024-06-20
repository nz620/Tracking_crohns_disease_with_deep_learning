import numpy as np
import SimpleITK as sitk
import os
from skimage import transform
from tqdm import tqdm
import random
import cv2
random.seed(2023)
from prompt_generator import get_centreline_points_from_file, centreline_prompt

# Path configuration
nii_path = "data/centreline_set/axial/img"
gt_path = "data/centreline_set/axial/seg"
centreline_path = "data/centreline_set/axial/centreline"
generated_centreilne_path = "data/centreline_set/axial/centreline_single_skeltonize"
npy_path = "data/centreline_set/axial/npy2023_5_folds_test"

# Constants
img_name_suffix = ".nii.gz"
gt_name_suffix = ".nii.gz"
centerline_name_suffix = " HASTE tracings"
generated_centreilne_name_suffix = ".txt"
num_folds = 5
image_size = 1024

for fold in range(num_folds):
    os.makedirs(os.path.join(npy_path, f"fold_{fold}", "gts"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, f"fold_{fold}", "imgs"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, f"fold_{fold}", "centreline"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, f"fold_{fold}", "centreline_generated"), exist_ok=True)

# List and filter image files
names = sorted(os.listdir(gt_path))
names = [name for name in names if os.path.exists(os.path.join(nii_path, name.replace(gt_name_suffix, img_name_suffix)))]
names = [name for name in names if os.path.exists(os.path.join(centreline_path, name.replace(gt_name_suffix, centerline_name_suffix)))]
names = [name for name in names if os.path.exists(os.path.join(generated_centreilne_path, name.replace(gt_name_suffix, generated_centreilne_name_suffix)))]
# Assign folds sequentially
fold_assignments = {}
fold_sizes = (len(names) + num_folds - 1) // num_folds  
print(names)
random.shuffle(names)
print(names)
for i, name in enumerate(names):
    fold = i % num_folds
    if fold not in fold_assignments:
        fold_assignments[fold] = []
    fold_assignments[fold].append(name)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Processing and saving logic
for fold, names_in_fold in fold_assignments.items():
    for name in tqdm(names_in_fold):
        image_name = name.replace(gt_name_suffix, img_name_suffix)
        gt_name = name
        centreline_name = name.replace(gt_name_suffix, centerline_name_suffix)
        centreline_points = get_centreline_points_from_file(os.path.join(centreline_path, centreline_name), percentage=20,keep_abnormal_pts=True, keep_normal_pts=True)
        generated_centreline_name = name.replace(gt_name_suffix, generated_centreilne_name_suffix)
        generated_centreline_points = get_centreline_points_from_file(os.path.join(generated_centreilne_path, generated_centreline_name), percentage=20,keep_abnormal_pts=True, keep_normal_pts=True)
        gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

        z_index=[]
        # find non-zero slices
        for i in range(len(gt_data_ori)):
            centreline_prompt_points = centreline_prompt(centreline_points,i)
            generated_centreilne_prompt_points = centreline_prompt(generated_centreline_points,i)
            if np.any(gt_data_ori[i]>0) and (len(centreline_prompt_points) > 0 or len(generated_centreilne_prompt_points) > 0):
                z_index.append(i)

        if len(z_index) > 0:
            # crop the ground truth with non-zero slices
            gt_roi = gt_data_ori[z_index, :, :]
            
            # load image and preprocess
            img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
            image_data = sitk.GetArrayFromImage(img_sitk)
            # nii preprocess start
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0
            image_data_pre = np.uint8(image_data_pre)
            for i in range(image_data_pre.shape[0]):
                image_data_pre[i] = clahe.apply(image_data_pre[i])

            fold_path = os.path.join(npy_path, f"fold_{fold}")
            for i in z_index:
                ori_h, ori_w = image_data_pre[i, :, :].shape
                scale_h, scale_w = image_size / ori_h, image_size / ori_w
                prev_idx = i - 1 if i - 1 in z_index else i
                next_idx = i + 1 if i + 1 in z_index else i
                
                img_prev = image_data_pre[prev_idx, :, :]
                img_next = image_data_pre[next_idx, :, :]
                img_i = image_data_pre[i, :, :]
                ## repeat a single channel to 3 channels
                # img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                img_3c = np.stack([img_prev, img_i, img_next], axis=-1)
                resize_img_skimg = transform.resize(
                    img_3c,
                    (image_size, image_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
                gt_i = gt_data_ori[i, :, :]
                resize_gt_skimg = transform.resize(
                    gt_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
                centreline_points_scaled = centreline_points.copy()
                centreline_points_scaled['x'] *= scale_h
                centreline_points_scaled['y'] *= scale_w
                cenreline_points_roi = centreline_points_scaled[centreline_points_scaled['z'] == i][['x', 'y']].to_numpy()
                generated_centreline_points_scaled = generated_centreline_points.copy()
                generated_centreline_points_scaled['x'] *= scale_h
                generated_centreline_points_scaled['y'] *= scale_w
                generated_cenreline_points_roi = generated_centreline_points_scaled[generated_centreline_points_scaled['z'] == i][['x', 'y']].to_numpy()
                
                np.save(os.path.join(fold_path, "imgs", f"{name.replace(gt_name_suffix, '')}_{str(i).zfill(3)}.npy"), resize_img_skimg_01)
                np.save(os.path.join(fold_path, "gts", f"{name.replace(gt_name_suffix, '')}_{str(i).zfill(3)}.npy"), resize_gt_skimg)
                np.save(os.path.join(fold_path, "centreline", f"{name.replace(gt_name_suffix, '')}_{str(i).zfill(3)}.npy"), cenreline_points_roi)
                np.save(os.path.join(fold_path, "centreline_generated", f"{name.replace(gt_name_suffix, '')}_{str(i).zfill(3)}.npy"), generated_cenreline_points_roi)
