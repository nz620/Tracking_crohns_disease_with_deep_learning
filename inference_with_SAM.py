import os 
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
from ori_sam import sam_model_registry, SamPredictor
import time 
import random
from PIL import Image
from preprocessor import Preprocessor
from dataloader import NpyDataset
import pandas as pd
from prompt_generator import generate_bounding_boxes,generate_negative_point_prompt,generate_point_prompt,centreline_prompt,get_centreline_points_from_file
from utils.display_helper import show_mask, show_points, show_box
parser = argparse.ArgumentParser(description='MRI Segmentation')
parser.add_argument('--mode', type=str, required=True, default='box',
                    help='Mode of operation: box, point, two_points, point_and_box, centreline')
parser.add_argument('--prepocess',type=bool,default=False,help='choose whetherPreprocess the image ')
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--image_size", type=int, default=1024, help="image_size")
parser.add_argument(
    "--sam_checkpoint", type=str, default="work_dir/SAM_MED2D_testing_coronal-20240411-0347/medsam_model_best.pth"
)
parser.add_argument("--task", type=str, default="SAM_MED2D", help="Test_Task")

args = parser.parse_args()

def read_mri_with_TI_mask(image_path,label_path):
    #### read the image and label and prepare for TI
    image = sitk.ReadImage(image_path,sitk.sitkFloat32)
    label = sitk.ReadImage(label_path)
    label_array = sitk.GetArrayFromImage(label)
    print('array loaded')
    relabel_array =  np.where((label_array == 1 ) | (label_array == 2) , 1, label_array)
    print('TI relabeled')
    relabel_array = np.where((label_array == 3) | (label_array == 4) | (label_array == 6), 0, relabel_array)
    return image,relabel_array

def adapt_image_to_model(image):
    #### adapt the image to the model
    output_image = []
    print("image shape:",image.shape)
    for i in range(image.shape[0]):
        image_rgb = np.stack((image[i],)*3, axis=-1)
        output_image.append(image_rgb)
    return output_image

def display_prediction_boxprompt(image, mask,prompt,gt,i,show=True):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image,cmap='gray')
    show_mask(mask, ax[0])
    show_box(prompt, ax[0])
    ax[0].set_title('Prediction')
    ax[1].imshow(image,cmap='gray')
    show_mask(gt, ax[1])
    show_box(prompt, ax[1])
    ax[1].set_title('Ground Truth')
    plt.savefig(os.path.join('finetune_SAM_result','A97 Axail slice' + str(i) +'(box prompt)'+ '.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def display_prediction_pointprompt(image, mask,prompt,gt,i,show=True):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image,cmap='gray')
    show_mask(mask, ax[0])
    show_points(prompt[0], prompt[1], ax[0])
    ax[0].set_title('Prediction')
    ax[1].imshow(image,cmap='gray')
    show_mask(gt, ax[1])
    show_points(prompt[0], prompt[1], ax[1])
    ax[1].set_title('Ground Truth')
    save_path = os.path.join(args.mode+'A3 Coronal' +args.task, 'slice'+str(i) +'(point prompt)'+ '.png')
    if not os.path.exists(args.mode+'A3 Coronal' +args.task):
        os.makedirs(args.mode+'A3 Coronal' +args.task)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def display_prediction_TwoPointsPrompt(image, mask,prompt,gt,i,show=True):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image,cmap='gray')
    show_mask(mask, ax[0])
    show_points(prompt[0], prompt[1], ax[0])
    ax[0].set_title('Prediction')
    ax[1].imshow(image,cmap='gray')
    show_mask(gt, ax[1])
    show_points(prompt[0], prompt[1], ax[1])
    ax[1].set_title('Ground Truth')
    plt.savefig(os.path.join('sam_testing_result_thres0.8/A101 Axial postive and negative point prompt','A101 Axial slice' + str(i) +'(two point prompt)'+ '.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

def display_prediction_PointandBoxPrompt(image, mask,prompt_box,prompt_point,gt,i,show=True):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image,cmap='gray')
    show_mask(mask, ax[0])
    show_points(prompt_point[0], prompt_point[1], ax[0])
    show_box(prompt_box, ax[0]) 
    ax[0].set_title('Prediction')
    ax[1].imshow(image,cmap='gray')
    show_mask(gt, ax[1])
    show_points(prompt_point[0], prompt_point[1], ax[1])
    show_box(prompt_box, ax[1]) 
    ax[1].set_title('Ground Truth')
    plt.savefig(os.path.join('sam_testing_result_thres0.8/A101 Axial box and point prompt','A101 Axial slice' + str(i) +'(box and point prompt)'+ '.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
        
def evaluation_dice(prediction,gt):
    #### calculate dice score
    TI_overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    prediction_uint8 = [np.squeeze(np.array(item).astype(np.uint8), axis=-1) for item in prediction]
    prediction_seg = sitk.GetImageFromArray(prediction_uint8)
    prediction_seg = sitk.Cast(prediction_seg, sitk.sitkUInt8)
    ground_truth_uint8=[np.array(item).astype(np.uint8)for item in gt]
    ground_truth_seg = sitk.GetImageFromArray(ground_truth_uint8)
    ground_truth_seg = sitk.Cast(ground_truth_seg, sitk.sitkUInt8) 
    TI_overlap_measures.Execute(prediction_seg==1, ground_truth_seg==1)
    TI_dice = TI_overlap_measures.GetDiceCoefficient()
    return TI_dice
    
    
if __name__ == '__main__':
    preprocess = Preprocessor()
    sam_checkpoint = args.sam_checkpoint
    print('model loading')
    model_type = "vit_h"
    print('model type:',model_type)
    sam = sam_model_registry[model_type](args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    sam.to(device)
    predictor = SamPredictor(sam)
    print('model loaded')
    
    #prepare data and prompt for segmenting TI area test
    image_path = 'data/centreline_set/coronal/img/A3 coronal.nii.gz'
    label_path =  'data/centreline_set/coronal/seg/A3 coronal.nii.gz'
    centreline_file = 'data/centreline_set/coronal/centreline/A3 coronal HASTE tracings'
    print('data loading')
    image, gt = read_mri_with_TI_mask(image_path,label_path)
    # image = preprocess.preprocess(image)
    image_array = sitk.GetArrayFromImage(image)
    
    image_input = adapt_image_to_model(image_array) 
    print('gt shape:',gt.shape)
    print('data loaded')
    prediction = []
    ground_truth = []
    time_start_total = time.time()
    
    #------------------segment with centreline prompt--------------------
    if args.mode == 'centreline':
        centreline_points = get_centreline_points_from_file(centreline_file,percentage=20)
        print('prompt generated') 
        for slice_num in range(len(gt)):
            centreline_prompt_points = centreline_prompt(centreline_points,slice_num)
            if np.any(gt[slice_num]>0) and len(centreline_prompt_points) > 0:
                time_start = time.time()
                print('Slice encoding')
                predictor.set_image(image_input[slice_num])
                print('Slice encoded')
                points_label = np.ones(len(centreline_prompt_points))
                print('making prediction for slice',slice_num)
                mask, _, _ = predictor.predict(
                        point_coords=centreline_prompt_points,
                        point_labels=points_label,
                        box=None,
                        multimask_output=False,
                        )
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1)
                print()
                prediction.append(mask_image)
                print('prediction made for slice',slice_num)
                ground_truth.append(gt[slice_num])
                display_prediction_pointprompt(image_input[slice_num][:,:,0], mask, [centreline_prompt_points,points_label], gt[slice_num],slice_num,show=False)
                time_end = time.time()
                print('time cost for each slice', time_end-time_start, 's')
                TI_dice_slice = evaluation_dice(mask_image,gt[slice_num])
                print('Dice score for TI slice:'+str(slice_num),TI_dice_slice)
            else:
                continue
        TI_dice = evaluation_dice(prediction,ground_truth)
        print('Dice score for TI:',TI_dice)
                
            
                
            
   #------------------segment with box prompt--------------------          
    if args.mode == 'box':
        box_prompts = generate_bounding_boxes(gt)
        print('prompt generated') 
        for i in range(len(box_prompts)):
            if box_prompts[i] is not None:
                time_start = time.time()
                print('Slice encoding')
                predictor.set_image(image_input[i])
                print('Slice encoded')
                box = box_prompts[i]
                print('making prediction for slice',i)
                mask, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False,
                        )
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1)
                prediction.append(mask_image)
                print('prediction made for slice',i)
                ground_truth.append(gt[i])
                display_prediction_boxprompt(image_input[i][:,:,0], mask, box, gt[i],i,show=False)
                time_end = time.time()
                print('time cost for each slice', time_end-time_start, 's')
                TI_dice_slice = evaluation_dice(mask_image,gt[i])
                print('Dice score for TI slice:'+str(i),TI_dice_slice)
        TI_dice = evaluation_dice(prediction,ground_truth)
        print('Dice score for TI:',TI_dice)
    #---------------------------------------------------------
    
    #------------------segment with point prompt------------------
    if args.mode == 'point':
        point_prompt = generate_point_prompt(gt)
        for i in range(len(point_prompt)):
            if point_prompt[i] is not None:
                time_start = time.time()
                print('Slice encoding')
                predictor.set_image(image_input[i])
                print('Slice encoded')
                point = np.array(point_prompt[i][0])
                print('point:',point)
                point_label = np.array(point_prompt[i][1])
                print('point label:',point_label)
                print('making prediction for slice',i)
                mask, _, _ = predictor.predict(
                        point_coords=point,
                        point_labels=point_label,
                        box=None,
                        multimask_output=False,
                        )
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1)
                prediction.append(mask_image)
                print('prediction made for slice',i)
                ground_truth.append(gt[i])
                display_prediction_pointprompt(image_input[i][:,:,0], mask, [point,point_label], gt[i],i,show=False)
                time_end = time.time()
                print('time cost for each slice', time_end-time_start, 's')
                TI_dice_slice = evaluation_dice(mask_image,gt[i])
                print('Dice score for TI slice:'+str(i),TI_dice_slice)
        TI_dice = evaluation_dice(prediction,ground_truth)
        print('Dice score for TI:',TI_dice)
    #--------------------------------------------------------------------------------------------------
        
    #------------------segment with one positive point prompt and one negative prompt------------------
    if args.mode == 'two-point':
        point_prompt = generate_point_prompt(gt)
        negative_point_prompts = generate_negative_point_prompt(gt)
        for i in range(len(point_prompt)):
            if point_prompt[i] is not None:
                time_start = time.time()
                print('Slice encoding')
                predictor.set_image(image_input[i])
                print('Slice encoded')
                point = np.vstack((point_prompt[i][0], negative_point_prompts[i][0]))
                print('point:',point)
                point_label = np.hstack((point_prompt[i][1], negative_point_prompts[i][1]))
                print('point label:',point_label)
                print('making prediction for slice',i)
                mask, _, _ = predictor.predict(
                        point_coords=point,
                        point_labels=point_label,
                        box=None,
                        multimask_output=False,
                        )
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1)
                prediction.append(mask_image)
                print('prediction made for slice',i)
                ground_truth.append(gt[i])
                display_prediction_TwoPointsPrompt(image_input[i][:,:,0], mask, [point,point_label], gt[i],i,show=False)
                time_end = time.time()
                print('time cost for each slice', time_end-time_start, 's')
                TI_dice_slice = evaluation_dice(mask_image,gt[i])
                print('Dice score for TI slice:'+str(i),TI_dice_slice)
        TI_dice = evaluation_dice(prediction,ground_truth)
        print('Dice score for TI:',TI_dice)
    #--------------------------------------------------------------
    
    #------------------segment with one positive point prompt and box prompt------------------
    if args.mode == 'boxpoint':
        point_prompt = generate_point_prompt(gt)
        box_prompts = generate_bounding_boxes(gt)
        for i in range(len(point_prompt)):
            if point_prompt[i] is not None:
                time_start = time.time()
                print('Slice encoding')
                predictor.set_image(image_input[i])
                print('Slice encoded')
                point = np.array(point_prompt[i][0])
                print('point:',point)
                point_label = np.array(point_prompt[i][1])
                print('point label:',point_label)
                box = box_prompts[i]
                print('box:',box)
                print('making prediction for slice',i)
                mask, _, _ = predictor.predict(
                        point_coords=point,
                        point_labels=point_label,
                        box=box,
                        multimask_output=False,
                        )
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1)
                prediction.append(mask_image)
                print('prediction made for slice',i)
                ground_truth.append(gt[i])
                display_prediction_PointandBoxPrompt(image_input[i][:,:,0], mask, box, [point,point_label], gt[i],i,show=False)
                time_end = time.time()
                print('time cost for each slice', time_end-time_start, 's')
                TI_dice_slice = evaluation_dice(mask_image,gt[i])
                print('Dice score for TI slice:'+str(i),TI_dice_slice)
        TI_dice = evaluation_dice(prediction,ground_truth)
        print('Dice score for TI:',TI_dice)
    #-------------------------------------------------------------------------
    
    #------------show the prompt and gt-----------------------
    # for i in range(len(box_prompts)):
    #     if box_prompts[i] is not None:
    #         box = box_prompts[i]
    #         print('making prediction for slice',i)
    #         fig = plt.figure(figsize=(10,10))
    #         show_box(box, plt.gca())
    #         show_mask(gt[i], plt.gca())
    #         plt.title(f"Prompt {i+1},", fontsize=18)
    #         plt.show()
    #---------------------------------------------------------
    
    #------------------show the prediction---------------------
    # print(len(prediction))  
    
    time_end_total = time.time()
    print('total time cost', time_end_total-time_start_total, 's')
