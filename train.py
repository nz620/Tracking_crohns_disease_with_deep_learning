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
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# # torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


join = os.path.join
# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/centreline_set/coronal/npy",
    help="path to training npy files; three subfolders: gts, imgs and centreline")
parser.add_argument("-task_name", type=str, default="samvitb_pretrain_img1024_smallimageadapter_64_aug2_newweakcentreline_mask")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("--sam_checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
parser.add_argument("--load_pretrain", type=bool, default=True, help="use wandb to monitor training")
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("-use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--image_size", type=int, default=1024, help="image_size")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
args = parser.parse_args()


run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir,run_id+args.task_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MedSAM(nn.Module):
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
medsam_model = MedSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder,
    prompt_encoder=sam_model.prompt_encoder,
).to(device)

def train():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    log_dir = os.path.join(model_save_path, "logs")
    writer = SummaryWriter(log_dir)

    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    trainable_params = [param for param in medsam_model.parameters() if param.requires_grad]
    
    # Initialize the optimizer with only trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print("Number of trainable parameters: ", sum(p.numel() for p in trainable_params))
    print("Number of trainable image encoder", sum(p.numel() for p in medsam_model.image_encoder.parameters() if p.requires_grad))
    print("Number of freeze image encoder", sum(p.numel() for p in medsam_model.image_encoder.parameters() if not p.requires_grad))
    print("Number of trainable mask decoder", sum(p.numel() for p in medsam_model.mask_decoder.parameters() if p.requires_grad))
    print("Number of freeze mask encoder", sum(p.numel() for p in medsam_model.mask_decoder.parameters() if not p.requires_grad))
    print("Number of trainable prompt encoder", sum(p.numel() for p in medsam_model.prompt_encoder.parameters() if p.requires_grad))
    print("Number of freeze prompt encoder", sum(p.numel() for p in medsam_model.prompt_encoder.parameters() if not p.requires_grad))
    seg_loss = FocalDiceloss_IoULoss()
    # dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    # ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    train_losses = []
    val_losses = []
    coronal_dataset = NpyDataset('data/centreline_set/coronal/npy',augment=True)
    coronal_dataset_validate = NpyDataset('data/centreline_set/coronal/npy/validation',augment=False)
    # axial_dataset = NpyDataset('data/centreline_set/axial/npy',augment=True)
    # axial_dataset_validate = NpyDataset('data/centreline_set/axial/npy/validation',augment=False)
    coronal_dataset_weak_centreline = NpyDataset('data/coronal/npy_single',augment=True)
    coronal_dataset_weak_centreline_validate = NpyDataset('data/coronal/npy_single/validation',augment=False)
    # Create DataLoaders for training and validation sets
    train_dataset = ConcatDataset([coronal_dataset, coronal_dataset_weak_centreline])
    valid_dataset = ConcatDataset([coronal_dataset_validate, coronal_dataset_weak_centreline_validate])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Number of training samples: ", len(train_dataloader))
    print("Number of validation samples: ", len(valid_dataloader))
    
    start_epoch = 0
    best_loss = 1e10
    patience = 7
    patience_counter = 0  
    
    for epoch in range(start_epoch, num_epochs):
        medsam_model.train()
        train_loss = 0
        for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            point_np = point.detach().cpu().numpy()
            point_label_np = point_label.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            medsam_pred,iou_pred= medsam_model(image, point_np,point_label_np)
            # loss = dice_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss = seg_loss(medsam_pred,gt2D, iou_pred)
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
        medsam_model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(valid_dataloader)):
                point_np = point.detach().cpu().numpy()
                point_label_np = point_label.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                medsam_pred,iou_pred = medsam_model(image, point_np,point_label_np)
                # loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss = seg_loss(medsam_pred,gt2D, iou_pred)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_dataloader)
            val_losses.append(avg_val_loss)
            print(
                    f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Val Loss: {avg_val_loss}'
                )
            writer.add_scalar('Loss/val', val_loss, epoch)
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  
            # Save the best model
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print(f"Epoch {epoch}: Validation loss improved to {avg_val_loss}. Saving model.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}: Validation loss did not improve. ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"Stopping training early at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
            break  # Stop the training loop
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Training and Validation Losses Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, "train_val_loss_plot.png"))
        plt.show()
        print("Training and Validation Losses Plot Saved")
    
    writer.close()
    
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = join(args.work_dir, run_id + args.task_name)
    # model_save_path = "work_dir/20240502-2111samvitb_pretrain_img1024_smallimageadapter_64_aug0_newweakcentreline_mask"
    os.makedirs(model_save_path, exist_ok=True)
    pre_save_path = join(model_save_path, "visual_test_results")
    os.makedirs(pre_save_path, exist_ok=True)
    
    best_checkpoint_path = join(model_save_path, "medsam_model_best.pth")
    # best_checkpoint_path = "work_dir/20240502-2111samvitb_pretrain_img1024_smallimageadapter_64_aug0_newweakcentreline_mask/medsam_model_best.pth"
    checkpoint = torch.load(best_checkpoint_path, map_location=device) 
    medsam_model.load_state_dict(checkpoint['model'])
    medsam_model.to(device)
    medsam_model.eval()
    coronal_dataset_test = NpyDataset('data/centreline_set/coronal/npy/test', augment=False)
    # axial_dataset_test = NpyDataset('data/centreline_set/axial/npy/test', augment=False)
    # test_dataset = ConcatDataset([coronal_dataset_test, axial_dataset_test])
    test_dataloader = DataLoader(coronal_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Number of test samples: ", len(test_dataloader))
    metrics = SegMetrics(metric_names=args.metrics)
    
    # with torch.no_grad():
    #     for step, (image, gt2D, point,point_label, img_names) in enumerate(tqdm(test_dataloader)):
    #         gt2D = _threshold(gt2D)
    #         image, gt2D = image.to(device), gt2D.to(device)
    #         preds, _ = medsam_model(image, point, point_label) 
    #         metrics.update(preds, gt2D, img_names)
    with torch.no_grad():
        for step, (image, gt2D, point, point_label, img_names) in enumerate(tqdm(test_dataloader)):
            image, gt2D = image.to(device), gt2D.to(device)
            preds, _ = medsam_model(image, point, point_label)  
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
    train()
    test()
    
    # base_path = 'data/coronal/npy_single'
    # valid_path = 'data/coronal/npy_single/validation'
    # test_path = 'data/coronal/npy_single/test'

    # folders = ['imgs', 'gts', 'centreline']

    # for folder in folders:
    #     src_folder = os.path.join(base_path, folder)
    #     valid_folder = os.path.join(valid_path, folder)
    #     test_folder = os.path.join(test_path, folder)
    #     os.makedirs(valid_folder, exist_ok=True) 
    #     os.makedirs(test_folder, exist_ok=True) 
        

    #     for filename in os.listdir(src_folder):
    #         if 'A104 ' in filename or 'A110 ' in filename or 'I1 ' in filename\
    #             or 'I11 ' in filename or 'I103 ' in filename or 'I113 ' in filename\
    #             or 'I16 ' in filename or 'I3 ' in filename :
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
