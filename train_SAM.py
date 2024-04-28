import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
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
from dataloader_augment import NpyDataset
from torch.utils.tensorboard import SummaryWriter
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6



# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/centreline_set/coronal/npy",
    help="path to training npy files; three subfolders: gts, imgs and centreline",
)
parser.add_argument("-task_name", type=str, default="sammed2d_pretrain_promptdecoder_maskdecoder_72augmentdata_realnoshuffle_batchsize2")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "--sam_checkpoint", type=str, default="work_dir/sam-med2d_b.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=1)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.00001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--image_size", type=int, default=256, help="image_size")
args = parser.parse_args()

# if args.use_wandb:
#     import wandb

#     wandb.login()
#     wandb.init(
#         project=args.task_name,
#         config={
#             "lr": args.lr,
#             "batch_size": args.batch_size,
#             "data_path": args.tr_npy_path,
#             "model_type": args.model_type,
#         },
#     )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir,run_id+args.task_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device(args.device)
# %% set up model


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
            param.requires_grad = True
            
        for n, value in self.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = False
            else:
                value.requires_grad = False

    def forward(self, image, point,point_label):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            point_torch = torch.as_tensor(point, dtype=torch.float32, device=image.device)
            point_label_torch = torch.as_tensor(point_label, dtype=torch.int, device=image.device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=[point_torch,point_label_torch],
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
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
        return ori_res_masks


def main():
    print('sam_med_2d_coronal_mask_decoder__augmentdata_realnoshuffle_vit_b')
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    log_dir = os.path.join(model_save_path, "logs")
    writer = SummaryWriter(log_dir)

    sam_model = sam_model_registry[args.model_type](args)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
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
    # print(
    #     "Number of image encoder and mask decoder parameters: ",
    #     sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    # )  # 93729252
    print("Number of trainable image encoder", sum(p.numel() for p in medsam_model.image_encoder.parameters() if p.requires_grad))
    print("Number of freeze image encoder", sum(p.numel() for p in medsam_model.image_encoder.parameters() if not p.requires_grad))
    print("Number of trainable mask decoder", sum(p.numel() for p in medsam_model.mask_decoder.parameters() if p.requires_grad))
    print("Number of freeze mask encoder", sum(p.numel() for p in medsam_model.mask_decoder.parameters() if not p.requires_grad))
    print("Number of trainable prompt encoder", sum(p.numel() for p in medsam_model.prompt_encoder.parameters() if p.requires_grad))
    print("Number of freeze prompt encoder", sum(p.numel() for p in medsam_model.prompt_encoder.parameters() if not p.requires_grad))
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = []
    val_losses = []
    best_loss = 1e10
    # axial_dataset = NpyDataset('data/centreline_set/axial/npy')
    coronal_dataset = NpyDataset('data/centreline_set/coronal/npy_256',augment=True)
    coronal_dataset_validate = NpyDataset('data/centreline_set/coronal/npy_256/validation',augment=False)
    coronal_dataset_weak_centreline = NpyDataset('data/coronal/npy_256',augment=True)
    train_dataset = []
    val_dataset = []
    
    # for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(axial_dataset)):
    #     match = re.match(r'^(A5|A97|I59)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
    #     if match:
    #         val_dataset.append((image, gt2D, point,point_label, image_name))
    #     else:
    #         train_dataset.append((image, gt2D, point,point_label, image_name))    
    # print('match axial')
    # for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(coronal_dataset)):
    #     match = re.match(r'^(A3|A79|I58)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
    #     if match:
    #         val_dataset.append((image, gt2D, point,point_label, image_name))
    #     else:
    #         train_dataset.append((image, gt2D, point,point_label, image_name))   
    # print('match coronal')     
    # full_dataset = NpyDataset(args.tr_npy_path)
    # train_dataset = []
    # val_dataset = []
    # for step, (image, gt2D, point,point_label, image_name) in enumerate(tqdm(full_dataset)):
    #     match = re.match(r'^(A3|A79|I58)[^\.]*\.npy$', image_name) # Axial A5|A97|I59  ##coronal A3 A79 I58
    #     if match:
    #         val_dataset.append((image, gt2D, point,point_label, image_name))
    #     else:
    #         train_dataset.append((image, gt2D, point,point_label, image_name))        
    # # Split dataset into training and validation
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    # train_dataset = ConcatDataset([coronal_dataset, coronal_dataset_weak_centreline])
    train_dataloader = DataLoader(coronal_dataset, batch_size=2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(coronal_dataset_validate, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Number of training samples: ", len(train_dataloader))
    print("Number of validation samples: ", len(test_dataloader))
    
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        medsam_model.train()
        train_loss = 0
        for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            point_np = point.detach().cpu().numpy()
            point_label_np = point_label.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            medsam_pred = medsam_model(image, point_np,point_label_np)
            # loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
            loss = seg_loss(medsam_pred, gt2D)
            loss.backward()
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
            for step, (image, gt2D, point,point_label, _) in enumerate(tqdm(test_dataloader)):
                point_np = point.detach().cpu().numpy()
                point_label_np = point_label.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                medsam_pred = medsam_model(image, point_np,point_label_np)
                # loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss = seg_loss(medsam_pred, gt2D)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(test_dataloader)
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
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
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
if __name__ == "__main__":
    main()
    # base_path = 'data/centreline_set/coronal/npy_256'
    # valid_path = 'data/centreline_set/coronal/npy_256/validation'

    # folders = ['imgs', 'gts', 'centreline']

    # for folder in folders:
    #     src_folder = os.path.join(base_path, folder)
    #     dest_folder = os.path.join(valid_path, folder)
    #     os.makedirs(dest_folder, exist_ok=True)  # Ensure the destination folder exists

    #     for filename in os.listdir(src_folder):
    #         if 'A3' in filename or 'A79' in filename or 'I58' in filename:
    #             src_file = os.path.join(src_folder, filename)
    #             dest_file = os.path.join(dest_folder, filename)
    #             shutil.move(src_file, dest_file)
    #             print(f'Moved {src_file} to {dest_file}')
