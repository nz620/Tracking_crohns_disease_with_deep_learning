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
parser.add_argument("--aug_num", type=int, default=1)
parser.add_argument("--complete_centreline",action='store_true', help="use completer centreline")
parser.add_argument("--not_complete_test",action='store_true', help="use not completed centreline in test set")
#model
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--sam_checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--encoder_adapter", action='store_false', help="use adapter")
parser.add_argument("--image_size", type=int, default=1024, help="image_size")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--reduce_ratio", type=int, default=16, help="adapter_emebeding to embed_dimension//reduce_ratio")
parser.add_argument("--adapter_mlp_ratio", type=float, default=0.15, help="adapter_mlp_ratio")
parser.add_argument("--parallel_cnn", action='store_true', help="use parallel branch" )
parser.add_argument("--train_prompt", action='store_true', help="train prompt")
parser.add_argument("--train_mask",action='store_true', help="train mask decoder")
# train
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
# prompt type 
parser.add_argument("--box_prompt", action='store_true', help = 'Use box prompt')
parser.add_argument("--point_prompt", action='store_true', help='Use point prompt')
#early stopping
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--min_delta", type=float, default=0.001)
parser.add_argument("--consecutive", type=int, default=7)
# Optimizer and scheduler parameters
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("--wp", type=int, default=3, help="warmup period")
#other                   
parser.add_argument("--use_wandb", action='store_true', help="use wandb to monitor training")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
parser.add_argument("--test_only", action='store_true', help="test the model")
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
        # freeze prompt encoder
        for param in self.mask_decoder.parameters():
            if args.train_mask:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        for param in self.prompt_encoder.parameters():
            if args.train_prompt:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        for n, value in self.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
                
            if args.parallel_cnn:
                if "parallel_cnn_branch" in n:
                    value.requires_grad = True

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
print(
    "Number of total parameters: ",
    sum(p.numel() for p in crohnsam_model.parameters()),
)  
print(
    "Number of trainable parameters: ",
    sum(p.numel() for p in crohnsam_model.parameters() if p.requires_grad),
) 
print("----------------------------------------------")
print("Number of trainable image encoder", sum(p.numel() for p in crohnsam_model.image_encoder.parameters() if p.requires_grad))
print("Number of freeze image encoder", sum(p.numel() for p in crohnsam_model.image_encoder.parameters() if not p.requires_grad))
print("Number of trainable mask decoder", sum(p.numel() for p in crohnsam_model.mask_decoder.parameters() if p.requires_grad))
print("Number of freeze mask encoder", sum(p.numel() for p in crohnsam_model.mask_decoder.parameters() if not p.requires_grad))
print("Number of trainable prompt encoder", sum(p.numel() for p in crohnsam_model.prompt_encoder.parameters() if p.requires_grad))
print("Number of freeze prompt encoder", sum(p.numel() for p in crohnsam_model.prompt_encoder.parameters() if not p.requires_grad))
print("----------------------------------------------")

def load_train_datasets(args,fold_index=0):
    """
    Load training, validation, and optionally test datasets based on provided arguments.
    """
    # Load main dataset
    train_datasets = []
    validate_dataset = []
    for i in range(5):
        if i == fold_index:
            if args.not_complete_test:
                main_dataset_fold = NpyDataset(join(args.true_npy_path,args.data_type,'npy2023_5_folds_test',f'fold_{i}'), augment=False, aug_num=args.aug_num, complete_centreline=False)
            else:
                main_dataset_fold = NpyDataset(join(args.true_npy_path,args.data_type,'npy2023_5_folds_test',f'fold_{i}'), augment=False, aug_num=args.aug_num, complete_centreline=args.complete_centreline)
            validate_dataset = main_dataset_fold
        else:
            main_dataset_fold = NpyDataset(join(args.true_npy_path,args.data_type,'npy2023_5_folds_test',f'fold_{i}'), augment=True, aug_num=args.aug_num, complete_centreline=args.complete_centreline)
            train_datasets.append(main_dataset_fold)
        print(f'Fold {i} is loaded')
        
    if args.include_weak:
        weak_dataset = NpyDataset(join(args.weak_npy_path,args.data_type,'npy2023_test'), augment=True, aug_num=args.aug_num)
        # weak_dataset_validate = NpyDataset(join(args.weak_npy_path,args.data_type,'npy_single','validation'), augment=True, aug_num=args.aug_num)
        train_datasets.append(weak_dataset)
    if args.include_pseudo:
        pseudo_dataset = NpyDataset(join(args.pseudo_npy_path,args.data_type,'npy2023_test'), augment=True, aug_num=args.aug_num)
        train_datasets.append(pseudo_dataset)
    train_dataset = ConcatDataset(train_datasets)
    # Create DataLoader for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, validate_dataloader

def lr_schedule(epoch, wp=10, mi=100, initial_lr=0.0001):
    if epoch <= wp:
        # Linear increase from 0 to initial_lr over the warmup period
        return initial_lr * (epoch / wp)
    elif wp < epoch <= wp + 5:
        # Constant learning rate for 5 epochs after warmup
        return initial_lr
    else:
        # Begin decay after wp + 10
        # Adjust the decay phase to start after wp + 10
        return initial_lr * (1 - ((epoch - (wp)) / (mi - (wp))))


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, consecutive=3):
        """
        patience: Number of epochs with no improvement after which training will be stopped.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        consecutive: Number of checks over which the rate of improvement is calculated.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.consecutive = consecutive
        self.patience_counter = 0
        self.best_loss = np.inf
        self.losses = []

    def __call__(self, current_loss):
        self.losses.append(current_loss)
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0  # reset counter if improvement
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print("Stopping training: no improvement in validation loss")
            return True

        if len(self.losses) > self.consecutive:
            # Calculate the rate of change
            deltas = [self.losses[i] - self.losses[i - 1] for i in range(-self.consecutive, 0)]
            mean_delta = np.mean(deltas)
            if abs(mean_delta) < self.min_delta:
                print("Stopping training: minimal improvement")
                return True

        return False

def train_and_test(fold):
    sam_model = sam_model_registry[args.model_type](args)
    crohnsam_model = CrohnSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    
    print("Testing fold: ", fold)
    if args.use_wandb:
        wandb.init(project="Tracking Crohn 5 fold",name=f'{args.task_name} fold {fold}', config=args)
        wandb.watch(crohnsam_model, log="all", log_freq=2000)
    crohnsam_model.train()
    trainable_params = [param for param in crohnsam_model.parameters() if param.requires_grad]
    
    # Initialize the optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # LambdaLR scheduler:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_schedule(epoch, wp=args.wp, mi=args.num_epochs, initial_lr=args.lr)
    )
    
    # seg_loss = FocalDiceloss_IoULoss() 
    seg_loss = CombinedLoss()
    num_epochs = args.num_epochs
    train_losses = []
    val_losses = []
    train_dataloader, validate_dataloader = load_train_datasets(args,fold)
    print("Number of training samples: ", len(train_dataloader))
    print("Number of validation samples: ", len(validate_dataloader))
    
    start_epoch = 0
    best_loss = 1e10
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, consecutive=args.consecutive)

    for epoch in range(start_epoch, num_epochs):
        crohnsam_model.train()
        train_loss = 0
        for step, (image, gt2D, point,point_label,box, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, gt2D, point,point_label,box = image.to(device), gt2D.to(device),point.to(device),point_label.to(device),box.to(device)
            crohnsam_pred,iou_pred= crohnsam_model(image, point,point_label,box)
            loss = seg_loss(crohnsam_pred,gt2D, iou_pred)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {avg_train_loss}'
        )
        crohnsam_model.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (image, gt2D, point,point_label,box, _) in enumerate(tqdm(validate_dataloader)):
                image, gt2D,point,point_label,box = image.to(device), gt2D.to(device),point.to(device),point_label.to(device),box.to(device)
                crohnsam_pred,iou_pred = crohnsam_model(image, point,point_label,box)
                # loss = seg_loss(crohnsam_pred, gt2D) + ce_loss(crohnsam_pred, gt2D.float())
                loss = seg_loss(crohnsam_pred,gt2D, iou_pred)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(validate_dataloader)
            val_losses.append(avg_val_loss)
            print(
                    f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Val Loss: {avg_val_loss}'
                )
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        ## save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = crohnsam_model.state_dict()    
            print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss}. Saving model.")
            checkpoint = {
                "model": crohnsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            checkpoint_path = join(model_save_path, f"crohnsam_model_best_fold{fold}.pth")
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved successfully at {checkpoint_path}")
            except Exception as e:
                print(f"Failed to save checkpoint at {checkpoint_path}: {e}")
        if early_stopping(avg_val_loss):
            print("Early stopping triggered.")
            break
        # if patience_counter >= patience:
        #     print(f"Stopping training early at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
        #     break  # Stop the training loop
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Training and Validation Losses Over Epochs - fold {fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, f"train_val_loss_plot - fold {fold}.png"))
        plt.close()
        print("Training and Validation Losses Plot Saved")
        scheduler.step()
        
    # testing the model 
    pre_save_path = join(model_save_path, f"visual_test_results - fold {fold}")
    os.makedirs(pre_save_path, exist_ok=True)
    crohnsam_model.load_state_dict(best_model)
    crohnsam_model.to(device)
    crohnsam_model.eval()
    print("Number of test samples: ", len(validate_dataloader))
    metrics = SegMetrics(metric_names=args.metrics,device=device)
    patient_data = {}
    with torch.no_grad():
        for step, (image, gt2D, point, point_label, box, img_names) in enumerate(tqdm(validate_dataloader)):
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
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].imshow(img)
            show_mask(smoothed_preds[0].cpu().numpy(), axs[0])
            if args.point_prompt :
             show_points(point[0].cpu().numpy(), point_label[0].cpu().numpy(), axs[0])
            if args.box_prompt :
                for i in range(box[0].shape[0]):
                    show_box(box[0][i,:].cpu().numpy(), axs[0])
            axs[0].axis("off")
            axs[0].set_title(f'prediction of {img_names[0]}')
            axs[1].imshow(img)
            show_mask(gt2D[0].cpu().numpy(), axs[1])
            if args.point_prompt :
             show_points(point[0].cpu().numpy(), point_label[0].cpu().numpy(), axs[1])
            if args.box_prompt :
                for i in range(box[0].shape[0]):
                    show_box(box[0][i,:].cpu().numpy(), axs[1])
            axs[1].axis("off")
            axs[1].set_title(f'ground truth of {img_names[0]}')
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
    
def main():
    print("Training Strategy: ", args.task_name)
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(model_save_path, os.path.basename(__file__)))

    all_fold_results = []
    # for fold_index in range(5):
    #     print(f"Starting training and testing for Fold {fold_index}")
    #     results = train_and_test(fold_index)
    #     all_fold_results.append(results)
    results = train_and_test(2)
    all_fold_results.append(results)
    average_metrics = calculate_average_metrics(all_fold_results)
    print("Average Metrics Across All Folds:", average_metrics)

    # Save average metrics to JSON file for later review or analysis.
    with open(os.path.join(model_save_path, "average_metrics.json"), 'w') as f:
        json.dump(average_metrics, f, indent=4)


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
    test_dataset = NpyDataset(join(args.true_npy_path,args.data_type,'npy2023_5_folds_test',f'fold_{fold}'), augment=False, aug_num=args.aug_num, complete_centreline=args.complete_centreline)
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
    if not args.test_only:
        main()
    else:
        test_main()
    
    
    