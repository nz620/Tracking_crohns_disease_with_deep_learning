"""
Given images, labels, and reference points, generate mask along the centreline.
Please follow the same dir to make the code work:
"./path" contains tracing (reference points) files: e.g. A1 coronal HASTE tracings.csv, A2 coronal HASTE tracings.csv, ...
"./cases/" contains images files: e.g. A1 coronal T2.nii, A2 coronal T2.nii, ...
"./meta/" contains files documenting image size and spacing: A1 coronal HASTE tracings_meta.txt, ...
"./masks/" contains output files: e.g. A1_coronal_mask_6_20.nii.gz, A2_coronal_mask_6_20.nii.gz, ...
"""

import os
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm.notebook import tqdm
import math

def compute_phy_distance(point1, point2, spacing):
    """
    point1 and point2 are 1*3 np array
    """
    dx = (point1[0] - point2[0]) * spacing[0]
    dy = (point1[1] - point2[1]) * spacing[1]
    dz = (point1[2] - point2[2]) * spacing[2]
    return math.sqrt((dx**2)+(dy**2)+(dz**2))

def get_mask(r, spacing, positions, imgsize):
    """
    Generate mask given the points along the centreline
    Remeber to swap axes
    """
    mask = np.zeros(imgsize)
    distance = np.ceil(np.array([r, r, r]) / np.array(spacing))
    for position in positions:
        for x in range(int(position[0]-distance[0]), int(position[0]+distance[0])):
            for y in range(int(position[1]-distance[1]), int(position[1]+distance[1])):
                for z in range(int(position[2]-distance[2]), int(position[2]+distance[2])):
                    if compute_phy_distance(np.array([x,y,z]), position, spacing) <= r:
                        mask[x][y][z] = 1
                    else:
                        pass
    numpy_mask = np.swapaxes(mask,0,2)
    return numpy_mask

def insert_point(pt1, pt2, spacing, dd=2):
    pt1_phy = pt1 * spacing
    pt2_phy = pt2 * spacing
    d = pt2_phy - pt1_phy
    t = math.ceil(compute_phy_distance(pt1, pt2, spacing) / dd) - 1
    
    insert_ = []
    interval = 1 / t

    for i in range(1, t+1, 1):
        xx = round((pt1_phy[0] + interval * d[0] * i) / spacing[0])
        yy = round((pt1_phy[1] + interval * d[1] * i) / spacing[1])
        zz = round((pt1_phy[2] + interval * d[2] * i) / spacing[2])

        insert_.append(np.array([xx, yy, zz]))
        
    return np.array(insert_)

def generate_mask(r, p):
    # loop through files
    for filename in tqdm(os.listdir('./paths')):
        casename = filename[0:-19]
        mask_path = "./masks/" + casename.replace(" ", "_") + "_mask_" + str(r) + "_" + str(p) + ".nii.gz"
        if not os.path.exists(mask_path):
            # generate mask if it doesn't exist
            img_path = "./cases/" + casename + " T2.nii"
            if os.path.exists(img_path) or os.path.exists(img_path + ".gz"):
                img = sitk.ReadImage(img_path)
                img_size = img.GetSize()
            else:    
                print("{} is not exist.".format(img_path))
                continue

            # read points
            path_df = pd.read_csv("./paths/" + filename)
            path_df.columns = ['x', 'y', 'z', 'xd', 'yd', 'zd']
            pix_co = path_df.iloc[:, 0:3].to_numpy()
            ratio = p / 100
            num_pts = int(pix_co.shape[0] * ratio)

            # read spacing and size
            meta_file_path = "./meta/" + casename + " HASTE tracings_meta.txt"
            f = open(meta_file_path, "r")
            temp = f.read().split(',')
            meta_info = [float(n) if "\n" not in n else float(n[0:-1]) for n in temp]
            case_spacing = meta_info[0:3]
            case_size = tuple([int(s) for s in meta_info[3:6]])
            if img_size != case_size:
                print("Case {}, size mismatch!".format(casename))
                continue
            
            terminal_ = pix_co[0:num_pts, :]
            # insert points
            for pt1, pt2 in zip(pix_co[0:num_pts-1, :], pix_co[1:num_pts, :]):
                distance = compute_phy_distance(pt1, pt2, case_spacing)
                if distance > 2:
                    inserted_pts = insert_point(pt1, pt2, case_spacing, dd=2)
                    terminal_ = np.concatenate((terminal_, inserted_pts), axis=0)
            
                
            # Compute mask
            np_mask = get_mask(r, tuple(case_spacing), terminal_, case_size)

            # save mask
            seg = sitk.GetImageFromArray(np_mask)
            seg.CopyInformation(img)
            mask_path = "./masks/" + casename.replace(" ", "_") + "_mask_" + str(r) + "_" + str(p) + ".nii.gz"
            sitk.WriteImage(seg, mask_path)
    else:
        pass
    
    print("Done!")

if __name__ == "__main__":
    generate_mask(6, 20)
