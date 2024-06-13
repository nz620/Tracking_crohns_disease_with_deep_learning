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
from dataloader_augment import NpyDataset
from torch.utils.tensorboard import SummaryWriter
import wandb
from torch.cuda.amp import GradScaler, autocast
# set seeds
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.empty_cache()

# # torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


join = os.path.join
#dataloading
parser = argparse.ArgumentParser()
parser.add_argument("--true_yy_path", type=str, default="data/centreline_set", help="path to training npy files; three subfolders: gts, imgs and centreline")
parser.add_argument('--include_weak', action='store_false', help='include weakly labelled data')
parser.add_argument("--weak_npy_path", type=str, default="data", help="path to weakly labelled npy files")
parser.add_argument("--task_name", type=str, default="samvitb_pretrain_img1024_smallimageadapter_32_aug2_newweakcentreline_mask")
parser.add_argument("--data_type", type=str, default="coronal")
parser.add_argument("--aug_num", type=int, default=1)
#model
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--sam_checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--encoder_adapter", action='store_false', help="use adapter")
parser.add_argument("--image_size", type=int, default=1024, help="image_size")
parser.add_argument("--work_dir", type=str, default="./work_dir")
parser.add_argument("--reduce_ratio", type=int, default=96, help="adapter_emebeding to embed_dimension//64")
# train
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
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
            param.requires_grad = True
            
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
            
        for n, value in self.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def forward(self, image, point,point_label):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            point_torch = torch.as_tensor(point, dtype=torch.float32, device=image.device)
            point_label_torch = torch.as_tensor(point_label, dtype=torch.int, device=image.device)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=[point_torch,point_label_torch],
                boxes=None,
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

def load_train_datasets(args):
    """
    Load training, validation, and optionally test datasets based on provided arguments.
    """
    # Load main dataset
    main_dataset = NpyDataset(join(args.true_npy_path,args.data_type,'npy'), augment=True, aug_num=args.aug_num)  # Example augmentation settings
    main_dataset_validate = NpyDataset(join(args.true_npy_path,args.data_type,'npy','validation'), augment=False)
    
    # Load weakly labelled dataset if specified
    if args.include_weak:
        weak_dataset = NpyDataset(join(args.weak_npy_path,args.data_type,'npy_single'), augment=True, aug_num=args.aug_num)
        weak_dataset_validate = NpyDataset(join(args.weak_npy_path,args.data_type,'npy_single','validation'), augment=False)
        
        # Combine datasets
        train_dataset = ConcatDataset([main_dataset, weak_dataset])
        validate_dataset = ConcatDataset([main_dataset_validate, weak_dataset_validate])
    else:
        train_dataset = main_dataset
        validate_dataset = main_dataset_validate
    
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

def train():

    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    log_dir = os.path.join(model_save_path, "logs")
    writer = SummaryWriter(log_dir)
    
    if args.use_wandb:
        wandb.init(project="Tracking Crohn",name=args.task_name, config=args)
        wandb.watch(crohnsam_model, log="all", log_freq=2000)
    crohnsam_model.train()
    scaler = GradScaler(growth_factor=1.01, backoff_factor=0.5)

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in crohnsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in crohnsam_model.parameters() if p.requires_grad),
    )  # 93729252

    trainable_params = [param for param in crohnsam_model.parameters() if param.requires_grad]
    
    # Initialize the optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # LambdaLR scheduler:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_schedule(epoch, wp=args.wp, mi=args.num_epochs, initial_lr=args.lr)
    )
    
    print("Number of trainable image encoder", sum(p.numel() for p in crohnsam_model.image_encoder.parameters() if p.requires_grad))
    print("Number of freeze image encoder", sum(p.numel() for p in crohnsam_model.image_encoder.parameters() if not p.requires_grad))
    print("Number of trainable mask decoder", sum(p.numel() for p in crohnsam_model.mask_decoder.parameters() if p.requires_grad))
    print("Number of freeze mask encoder", sum(p.numel() for p in crohnsam_model.mask_decoder.parameters() if not p.requires_grad))
    print("Number of trainable prompt encoder", sum(p.numel() for p in crohnsam_model.prompt_encoder.parameters() if p.requires_grad))
    print("Number of freeze prompt encoder", sum(p.numel() for p in crohnsam_model.prompt_encoder.parameters() if not p.requires_grad))
    seg_loss = FocalDiceloss_IoULoss()
    num_epochs = args.num_epochs
    train_losses = []
    val_losses = []
    train_dataloader, validate_dataloader = load_train_datasets(args)
    print("Number of training samples: ", len(train_dataloader))
    print("Number of validation samples: ", len(validate_dataloader))
    
    start_epoch = 0
    best_loss = 1e10
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, consecutive=args.consecutive)

    for epoch in range(start_epoch, num_epochs):
        crohnsam_model.train()
        train_loss = 0
        for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            # with autocast():  # Enable autocast for the forward pass
            #     images, gt2Ds = image.to(device), gt2D.to(device)
            #     crohnsam_pred, iou_pred = crohnsam_model(images, point, point_label)
            #     loss = seg_loss(crohnsam_pred, gt2Ds, iou_pred)
            # scaler.scale(loss).backward()  # Scale loss and call backward()
            # scaler.step(optimizer)  # Safe optimizer step
            # scaler.update()  # Update the scaler
            point_np = point.detach().cpu().numpy()
            point_label_np = point_label.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            crohnsam_pred,iou_pred= crohnsam_model(image, point_np,point_label_np)
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
        writer.add_scalar('Loss/train', train_loss, epoch)
        crohnsam_model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(validate_dataloader)):
                point_np = point.detach().cpu().numpy()
                point_label_np = point_label.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                crohnsam_pred,iou_pred = crohnsam_model(image, point_np,point_label_np)
                # loss = seg_loss(crohnsam_pred, gt2D) + ce_loss(crohnsam_pred, gt2D.float())
                loss = seg_loss(crohnsam_pred,gt2D, iou_pred)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(validate_dataloader)
            val_losses.append(avg_val_loss)
            print(
                    f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Val Loss: {avg_val_loss}'
                )
            writer.add_scalar('Loss/val', val_loss, epoch)
        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        ## save the latest model
        checkpoint = {
            "model": crohnsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "crohnsam_model_latest.pth"))
        ## save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  
            # Save the best model
            checkpoint = {
                "model": crohnsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "crohnsam_model_best.pth"))
            print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss}. Saving model.")
        # else:
        #     patience_counter += 1
        #     print(f"Epoch {epoch}: Validation loss did not improve. ({patience_counter}/{patience})")
        if early_stopping(avg_val_loss):
            print("Early stopping triggered.")
            break
        # if patience_counter >= patience:
        #     print(f"Stopping training early at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
        #     break  # Stop the training loop
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Training and Validation Losses Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, "train_val_loss_plot.png"))
        plt.close()
        print("Training and Validation Losses Plot Saved")
        scheduler.step()
    if args.use_wandb:
        wandb.finish()
    writer.close()
    
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = join(args.work_dir, run_id + args.task_name)
    # model_save_path = "work_dir/20240502-2111samvitb_pretrain_img1024_smallimageadapter_64_aug0_newweakcentreline_mask"
    os.makedirs(model_save_path, exist_ok=True)
    pre_save_path = join(model_save_path, "visual_test_results")
    os.makedirs(pre_save_path, exist_ok=True)
    
    best_checkpoint_path = join(model_save_path, "crohnsam_model_best.pth")
    # best_checkpoint_path = "work_dir/20240502-2111samvitb_pretrain_img1024_smallimageadapter_64_aug0_newweakcentreline_mask/crohnsam_model_best.pth"
    checkpoint = torch.load(best_checkpoint_path, map_location=device) 
    crohnsam_model.load_state_dict(checkpoint['model'])
    crohnsam_model.to(device)
    crohnsam_model.eval()
    
    test_dataset = NpyDataset(join(args.true_npy_path,args.data_type,'npy','test'), augment=False)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Number of test samples: ", len(test_dataloader))
    metrics = SegMetrics(metric_names=args.metrics)

    with torch.no_grad():
        for step, (image, gt2D, point, point_label, img_names) in enumerate(tqdm(test_dataloader)):
            image, gt2D = image.to(device), gt2D.to(device)
            preds, _ = crohnsam_model(image, point, point_label)  
            preds = preds.squeeze(1)  
            preds_np = preds.cpu().detach().numpy()  
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
            smoothed_preds_np = np.array([cv2.morphologyEx(cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel) for pred in preds_np])
            smoothed_preds = torch.tensor(smoothed_preds_np, dtype=torch.float32, device=device).unsqueeze(1) 
            
            metrics.update(smoothed_preds, gt2D, img_names)
            
            img = image[0].cpu().permute(1, 2, 0).numpy()
            img_norm = img /img.max()
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs[0].imshow(img_norm)
            show_mask(smoothed_preds[0].cpu().numpy(), axs[0])
            show_points(point[0].cpu().numpy(), point_label[0].numpy(), axs[0])
            axs[0].axis("off")
            axs[0].set_title(f'prediction of {img_names[0]}')
            axs[1].imshow(img_norm)
            show_mask(gt2D[0].cpu().numpy(), axs[1])
            show_points(point[0].cpu().numpy(), point_label[0].numpy(), axs[1])
            axs[1].axis("off")
            axs[1].set_title(f'ground truth of {img_names[0]}')
            plt.savefig(os.path.join(pre_save_path,  img_names[0] + ' prediction.png'), dpi=200, bbox_inches='tight')
            plt.close()

    overall_metrics, individual_results = metrics.compute()
    results = {
        "individual_metrics": individual_results,
        "overall_metrics": overall_metrics
    }
    
    with open(os.path.join(model_save_path, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Test Results saved to", os.path.join(model_save_path, 'test_metric.json'))
    
    
if __name__ == "__main__":
    # train()
    test()
    # base_path = 'data/axial/npy_single'
    # valid_path = 'data/axial/npy_single/validation'
    # test_path = 'data/axial/npy_single/test'

    # folders = ['imgs', 'gts', 'centreline']

    # for folder in folders:
    #     src_folder = os.path.join(base_path, folder)
    #     valid_folder = os.path.join(valid_path, folder)
    #     test_folder = os.path.join(test_path, folder)
    #     os.makedirs(valid_folder, exist_ok=True) 
    #     os.makedirs(test_folder, exist_ok=True) 
        

    #     for filename in os.listdir(src_folder):
    #         if 'A101 ' in filename or 'A107 ' in filename or 'I114 ' in filename\
    #             or 'I102 ' in filename or 'I108 ' in filename or 'A113 ' in filename\
    #             or 'I120 ' in filename :
    #             src_file = os.path.join(src_folder, filename)
    #             valid_file = os.path.join(valid_folder, filename)
    #             shutil.move(src_file, valid_file)
    #             print(f'Moved {src_file} to {valid_file}')
    #         # if 'A101' in filename or 'A107' in filename or 'A112' in filename\
    #         #     or 'I5' in filename or 'I14' in filename or 'I105' in filename\
    #         #     or 'I118' in filename or 'I120' in filename or 'I107' in filename:
    #         #     src_file = os.path.join(src_folder, filename)
    #         #     test_file = os.path.join(test_folder, filename)
    #         #     shutil.move(src_file, test_file)
    #         #     print(f'Moved {src_file} to {test_file}')
