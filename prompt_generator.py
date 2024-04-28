import os 
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
from segment_anything import sam_model_registry, SamPredictor
import time 
import random
from PIL import Image
import pandas as pd
import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path



def generate_bounding_boxes(relabel_array):
    #### generate bounding boxes for each slice
    bounding_boxes = []
    num_slices = relabel_array.shape[0]

    for i in range(num_slices):
        slice_relabel = relabel_array[i]

        # Check if the slice contains any relevant labels
        if np.any(slice_relabel):
            coords = np.argwhere(slice_relabel==1)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            box = [x0, y0, x1, y1]
            w, h = box[2] - box[0], box[3] - box[1]
            box = [x0-w*0.5, y0-h*0.5, x1+w*0.5, y1+h*0.5]
            bounding_boxes.append(np.array(box))
        else:
            bounding_boxes.append(None)  # Append None or appropriate value if no relevant label in the slice

    return bounding_boxes

def generate_point_prompt(relabel_array):
    points = []
    num_slices = relabel_array.shape[0]
    for i in range(num_slices):
        slice_relabel = relabel_array[i]
        if np.any(slice_relabel):
            coords = np.argwhere(slice_relabel==1)
            y,x= random.choice(coords)
            points.append([[np.array([x,y])],[1]])
        else:
            points.append(None)  # Append None or appropriate value if no relevant label in the slice

    return points

def generate_negative_point_prompt(relabel_array):
    points = []
    num_slices = relabel_array.shape[0]
    for i in range(num_slices):
        slice_relabel = relabel_array[i]
        if np.any(slice_relabel):
            coords = np.argwhere(slice_relabel==1)
            y,x= random.choice(coords)
            points.append([[np.array([x+random.randint(25, 50),y+random.randint(25, 50)])],[0]])
        else:
            points.append(None)  # Append None or appropriate value if no relevant label in the slice

    return points

def get_centreline_points_from_file(file_path, percentage=20, keep_normal_pts=True, keep_abnormal_pts=True):
    """
    Get a list of points [(x, y, z), ...] from an XML traces file provided by the clinicians.
    :param xml_file_path: Path to the XML file
    :param percentage: Percentage of the centreline to take into account (T.I.=20%)
    :param keep_normal_pts: Whether to keep the colon co-ordinates classified as being a normal region
    :param keep_abnormal_pts: Whether to keep the colon co-ordinates classified as being an abnormal region
    :return: List of centreline points in the format [(x, y, z), ...]
    """
    file_path = Path(file_path)
    out = []
    if file_path.suffix == '.txt':
        with open(file_path, 'r') as f:
           for line in f:
                parts = line.strip().split()
                slice_number = int(parts[0])
                x = int(parts[2])
                y = int(parts[1])
                out.append((x, y, slice_number))
        df = pd.DataFrame(out, columns=['x', 'y', 'z'])
        return df
    
    else:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for path in root:
            if 'name' not in path.attrib:
                continue
            
            # Check if the co-ordinate lies on an abnormal region of the colon
            is_abnormal_path = re.search('abnormal', path.attrib['name'], re.IGNORECASE)
            is_normal_path = not is_abnormal_path

            if is_abnormal_path and (not keep_abnormal_pts):
                continue

            if is_normal_path and (not keep_normal_pts):
                continue

            for point in path:
                attr = point.attrib
                out.append((int(attr['x']), int(attr['y']), int(attr['z'])))
        size = int(math.ceil(len(out) * percentage / 100))
        df = pd.DataFrame(out[:size], columns=['x', 'y', 'z'])
        return df
    
def centreline_prompt(df,slice_num):
    slice_points = df[df['z'] == slice_num][['x', 'y']].to_numpy()
    return slice_points

    
if __name__ == '__main__':
   pass
    