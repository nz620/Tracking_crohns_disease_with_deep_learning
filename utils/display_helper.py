import numpy as np
import matplotlib.pyplot as plt
import os

def show_mask(mask, ax, gt=False):
    if gt:
        color = np.array([30/ 255, 252 / 255, 30 / 255, 0.3])
    else:
        color = np.array([251 / 255,  25/ 255, 30 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if gt:
        ax.imshow(mask_image) 
    else:
        ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=60):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='.', s=marker_size)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size)
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
