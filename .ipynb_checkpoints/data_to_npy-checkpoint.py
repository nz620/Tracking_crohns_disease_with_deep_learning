import numpy as np
import SimpleITK as sitk
import os
join = os.path.join
from skimage import transform,exposure
from tqdm import tqdm
import cc3d
from prompt_generator import get_centreline_points_from_file,centreline_prompt

# convert nii image to npz files, including original image and corresponding masks

img_name_suffix = ".nii.gz"
gt_name_suffix = ".nii.gz"
centerline_name_suffix = ".txt"

nii_path = "data/coronal/img"  # path to the nii images
gt_path = "data/coronal/seg"  # path to the ground truth
centreline_path = "data/coronal/centreline_single" # path to the centreline
npy_path = "data/coronal/npy2023"
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)
os.makedirs(join(npy_path, "centreline"), exist_ok=True)

image_size = 1024
print(f"image size {image_size=}")

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")
names = [
    name
    for name in names  
    if os.path.exists(join(centreline_path, name.split(gt_name_suffix)[0] + centerline_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")


# %% save preprocessed images and masks as npz files
for name in tqdm(names): 
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    centreline_name = name.split(gt_name_suffix)[0] + centerline_name_suffix
    centreline_points = get_centreline_points_from_file(join(centreline_path,centreline_name),percentage=20, keep_normal_pts=True, keep_abnormal_pts=True)
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # exclude the objects with less than 1000 pixels in 3D
    # gt_data_ori = cc3d.dust(
    #     gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    # )
    z_index=[]
    # find non-zero slices
    for i in range(len(gt_data_ori)):
        centreline_prompt_points = centreline_prompt(centreline_points,i)
        if np.any(gt_data_ori[i]>0) and len(centreline_prompt_points) > 0:
            z_index.append(i)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
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

        # Histogram Equalization
        image_data_pre = exposure.equalize_hist(image_data_pre)
        
        # Standardization (Z-score normalization)
        image_data_pre = (image_data_pre - np.mean(image_data_pre)) / np.std(image_data_pre)
        
        # Clip to range [0, 1]
        image_data_pre = np.clip(image_data_pre, 0, 1)

        # Convert to uint8
        image_data_pre = np.uint8(image_data_pre * 255)
        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(join(npy_path, gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # save the image and ground truth as nii files for sanity check;
        # they can be remove
        # save the each CT image as npy file
        for i in z_index:
            ori_h, ori_w = image_data_pre[i, :, :].shape
            scale_h, scale_w = image_size / ori_h, image_size / ori_w
            img_i = image_data_pre[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
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
            np.save(
                join(
                    npy_path,
                    "imgs",
                    gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )
            np.save(
                join(
                    npy_path,
                    "gts",
                    gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )
            np.save(
                join(
                    npy_path,
                    "centreline",
                    gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                cenreline_points_roi,
            )