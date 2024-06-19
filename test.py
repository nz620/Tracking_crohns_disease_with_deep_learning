import json
import numpy as np
import matplotlib.pyplot as plt
import os
from loss import CombinedLoss
from metrics import SegMetrics, _threshold,_list_tensor
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader,ConcatDataset
from sam_med2d import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
from utils.display_helper import show_mask, show_points, show_box
import cv2
from dataloader_augment import NpyDataset
import wandb
import SimpleITK as sitk

# set seeds
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.empty_cache()


os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


join = os.path.join
#dataloading
parser = argparse.ArgumentParser()
parser.add_argument("--true_npy_path", type=str, default="data/centreline_set", help="path to training npy files; three subfolders: gts, imgs and centreline")
parser.add_argument('--include_weak', action='store_false', help='include weakly labelled data')
parser.add_argument("--weak_npy_path", type=str, default="data", help="path to weakly labelled npy files")
parser.add_argument('--include_pseudo', action='store_true', help='include pseudo labelled data')
parser.add_argument("--pseudo_npy_path", type=str, default="data/pseudo_data", help="path to pseudo labelled npy files")
parser.add_argument("--task_name", type=str, default="samvitb_pretrain_img1024_smallimageadapter_32_aug2_newweakcentreline_mask")
parser.add_argument("--data_type", type=str, default="coronal")
parser.add_argument("--complete_centreline",action='store_true', help="use completer centreline")
#model
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--sam_checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--encoder_adapter", action='store_false', help="use adapter")
parser.add_argument("--image_size", type=int, default=1024, help="image_size")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--reduce_ratio", type=int, default=16, help="adapter_emebeding to embed_dimension//reduce_ratio")
parser.add_argument("--adapter_mlp_ratio", type=float, default=0.15, help="adapter_mlp_ratio")
# prompt type 
parser.add_argument("--box_prompt", action='store_true', help = 'Use box prompt')
parser.add_argument("--point_prompt", action='store_true', help='Use point prompt')
#other                   
parser.add_argument("--use_wandb", action='store_true', help="use wandb to monitor training")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
args = parser.parse_args()


run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir,run_id+args.task_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CrohnSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
    def forward(self, image, point,point_label,box):
        image_embedding = self.image_encoder(image)
        if args.point_prompt:
            point_prompt = [point,point_label]
        else: 
            point_prompt = None
        if args.box_prompt:
            box_prompt = box
        else:
            box_prompt = None
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_prompt,
                boxes=box_prompt,
                masks=None,
            )
        low_res_masks, predict_iou= self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks, predict_iou
    
sam_model = sam_model_registry[args.model_type](args)
crohnsam_model = CrohnSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)

def calculate_average_metrics(all_fold_results):
    """Calculate average and standard deviation of metrics across all folds."""
    if not all_fold_results:
        return {}

    average_metrics = {}
    # Assume all folds have the same set of metrics for simplification
    metrics_keys = all_fold_results[0]['overall_metrics'].keys()  
    
    for metric in metrics_keys:
        average_metrics[metric] = {
            'mean_across_folds': [],
            'std_across_folds': [],
        }
        
        # Collect all patient averages for the current metric across all folds
        all_fold_averages = [fold['overall_metrics'][metric]['mean'] for fold in all_fold_results]
        
        # Compute the mean and standard deviation across all patient averages
        mean_across_folds = np.mean(all_fold_averages)
        std_across_folds = np.std(all_fold_averages)
        
    
        average_metrics[metric]['mean_across_folds'] = mean_across_folds
        average_metrics[metric]['std_across_folds'] = std_across_folds

    return average_metrics

def test_volume (fold,checkpoint):
    sam_model = sam_model_registry[args.model_type](args)
    with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu") 
    sam_model.load_state_dict(state_dict['model'], False)
    print("loaded model:",checkpoint)
    crohnsam_model = CrohnSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    
    pre_save_path = join(model_save_path, f"visual_test_results - fold {fold}")
    os.makedirs(pre_save_path, exist_ok=True)
    crohnsam_model.to(device)
    crohnsam_model.eval()
    test_dataset = NpyDataset(join(args.true_npy_path,args.data_type,'npy2023_5_folds_test',f'fold_{fold}'), augment=False, aug_num=0, complete_centreline=args.complete_centreline)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Number of test samples: ", len(test_dataloader))
    metrics = SegMetrics(metric_names=args.metrics,device=device)
    patient_data = {}
    with torch.no_grad():
        for step, (image, gt2D, point, point_label, box, img_names) in enumerate(tqdm(test_dataloader)):
            patient_id = img_names[0].split(' ')[0]
            image, gt2D,point,point_label,box = image.to(device), gt2D.to(device),point.to(device),point_label.to(device),box.to(device)
            preds, _ = crohnsam_model(image, point, point_label,box)  
            preds,gt2D = _list_tensor(preds,gt2D)
            preds = _threshold(preds, threshold=0.5)
            preds = preds.squeeze(1)  
            preds_np = preds.cpu().detach().numpy()  
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) 
            smoothed_preds_np = np.array([cv2.morphologyEx(cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel) for pred in preds_np])
            smoothed_preds = torch.tensor(smoothed_preds_np, dtype=torch.float32, device=device).unsqueeze(1) 
            img = image[0].cpu().permute(1, 2, 0).numpy()
            img_i =img[:,:,1]
            img = np.repeat(img_i[:, :, None], 3, axis=-1)
            # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            # axs[0].imshow(img)
            # show_mask(smoothed_preds[0].cpu().numpy(), axs[0])
            # if args.point_prompt :
            #  show_points(point[0].cpu().numpy(), point_label[0].cpu().numpy(), axs[0])
            # if args.box_prompt :
            #     for i in range(box[0].shape[0]):
            #         show_box(box[0][i,:].cpu().numpy(), axs[0])
            # axs[0].axis("off")
            # axs[0].set_title(f'prediction of {img_names[0]}')
            # axs[1].imshow(img)
            # show_mask(gt2D[0].cpu().numpy(), axs[1])
            # if args.point_prompt :
            #  show_points(point[0].cpu().numpy(), point_label[0].cpu().numpy(), axs[1])
            # if args.box_prompt :
            #     for i in range(box[0].shape[0]):
            #         show_box(box[0][i,:].cpu().numpy(), axs[1])
            # axs[1].axis("off")
            # axs[1].set_title(f'ground truth of {img_names[0]}')
            # plt.savefig(os.path.join(pre_save_path,  img_names[0] + ' prediction.png'), dpi=200, bbox_inches='tight')
            # plt.close()
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img)  
            show_mask(smoothed_preds[0].cpu().numpy(), ax)
            show_mask(gt2D[0].cpu().numpy(), ax,gt=True)
            # if args.point_prompt:
            #     show_points(point[0].cpu().numpy(), point_label[0].cpu().numpy(), ax)
            if args.box_prompt:
                for i in range(box[0].shape[0]):
                    show_box(box[0][i, :].cpu().numpy(), ax)
            ax.axis("off")
            ax.set_title(f'Prediction and Ground Truth of {img_names[0]}')
            plt.savefig(os.path.join(pre_save_path,  img_names[0] + ' prediction.png'), dpi=200, bbox_inches='tight')
            plt.close()
            #pred.shape = 1x1xHxW, gt2D.shape = 1x1xHxW
            # Aggregate predictions by patient
            if patient_id not in patient_data:
                patient_data[patient_id] = {'preds': [], 'gts': [], 'names': []}
            patient_data[patient_id]['preds'].append(smoothed_preds.squeeze(1))
            patient_data[patient_id]['gts'].append(gt2D.squeeze(1))
            patient_data[patient_id]['names'].append(img_names[0])
        
        for patient_id, data in patient_data.items():
            all_preds = torch.cat(data['preds'], dim=0) #shape is NxHxW
            all_gts = torch.cat(data['gts'], dim=0)#shape is Nx1xHxW  
            sitk.WriteImage(sitk.GetImageFromArray(all_preds.cpu().numpy()),os.path.join(pre_save_path,  patient_id + ' prediction.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(all_gts.cpu().numpy()),os.path.join(pre_save_path,  patient_id + ' ground_truth.nii.gz'))
            metrics.update(all_preds, all_gts, patient_id)
    overall_metrics, individual_results = metrics.compute()
    results = {
        "individual_metrics": individual_results,
        "overall_metrics": overall_metrics
    }
    with open(os.path.join(model_save_path, f"test_metrics_fold{fold}.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Test Results saved to", os.path.join(model_save_path, f"test_metrics_fold{fold}.json"))
    
    return results
    
def test_main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(model_save_path, os.path.basename(__file__)))

    all_fold_results = []
    
    for fold_index in range(5):
        print(f"Starting training and testing for Fold {fold_index}")
        results = test_volume(fold_index,f'work_dir/20240608-1741final_model_axial_gt_centreline/crohnsam_model_best_fold{fold_index}.pth')
        all_fold_results.append(results)
    # results = test_volume(2,'work_dir/20240606-07115fold_coronal_complete_centreline_pointprompt_direct_adapter_after_attention_skip_adapter_mlp_ablatio_adapter_mlp_ratio0.55/crohnsam_model_best_fold2.pth')
    all_fold_results.append(results)
    

    average_metrics = calculate_average_metrics(all_fold_results)
    print("Average Metrics Across All Folds:", average_metrics)

    with open(os.path.join(model_save_path, "average_metrics.json"), 'w') as f:
        json.dump(average_metrics, f, indent=4)
        
        
if __name__ == "__main__":
    test_main()