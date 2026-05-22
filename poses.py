from aux_functions import Get_Args, Save_In_Out_Target_Images
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from CapLayer import CapLayer
import torch
import torch.nn.functional as F
import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def BatchShift(imbatch: torch.Tensor, dx, padding_mode_sift, device):

    B, C, H, W = imbatch.shape
    R = torch.zeros(size=(B,2), device=device, dtype=torch.float32)
    R[:,0] = dx

    dx_norm = R[:, 0] / W 
    #dy_norm = R[:, 1] / H 

    theta = torch.zeros(B, 2, 3, device=device)

    theta[:, 0, 0] = 1.0         # escala X
    theta[:, 1, 1] = 1.0         # escala Y
    theta[:, 0, 2] = dx_norm      # translação X
    theta[:, 1, 2] = 0    # translação Y

    grid = F.affine_grid(theta, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear', 
                             padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R


def Save_In_Out_Target_Images(inp, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, type):

    batch = torch.cat([inp, target, out], dim=3)
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5)
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    

    caminho = os.path.join(RESULTS_DIR_IN_OUT_TARGET_IMAGES, f'{type}_batch_{i:04d}.png')
    plt.imsave(caminho, img)

def Save_Plot_Poses_LessOriginal_MoreOriginal(poses_combined_less, poses_combined_more, POSES_DIR):

    plt.figure(figsize=(11, 11))

    # Data Processing - less (-3px)
    x_less = poses_combined_less[:, 0]
    x_original = poses_combined_less[:, 1]
    # Least Squares 
    m_less, b_less = np.polyfit(x_less, x_original, 1)
    y_pred_less = m_less * x_less + b_less
    # r2_score measures how well the predicted values (y_pred_less) match the original values (y_original). 
    # An R² of 1 means perfect prediction, 
    # In this context, a higher R² would suggest that the poses with the -3px shift are more closely aligned with the original poses, 
    # indicating better equivariance to that transformation.
    r2_less = r2_score(x_original, y_pred_less)
    plt.scatter(x_less, x_original, alpha=0.2, s=5, color='tab:green', label='x -> x_displaced; y -> x_original')    
    plt.plot(x_less, y_pred_less, color='darkgreen', lw=2,
        label=f'Reta Less (R²={r2_less:.4f})')
    
    # Data Processing - more (+3px)
    x_more = poses_combined_more[:, 0]
    x_original_v1 = poses_combined_more[:, 1]
    # Least Squares
    m_more, b_more = np.polyfit(x_more, x_original_v1, 1)
    y_pred_more = m_more * x_more + b_more
    # r2_score measures how well the predicted values (y_pred_less) match the original values (y_original). 
    # An R² of 1 means perfect prediction, 
    # In this context, a higher R² would suggest that the poses with the -3px shift are more closely aligned with the original poses, 
    # indicating better equivariance to that transformation.
    r2_more = r2_score(x_original_v1, y_pred_more)
    plt.scatter(x_more, x_original_v1, alpha=0.2, s=5, color='tab:blue', label='x -> x_displaced_; y -> x_original')
    plt.plot(x_more, y_pred_more, color='darkblue', lw=2,
        label=f'Reta More (R²={r2_more:.4f})')
        
    # Plot Configurations
    plt.xlabel('Pose — Displaced Image (Less or More)')
    plt.ylabel('Pose — Original Image')
    plt.title('Outputs of module "study_capsule_index" before and after shift')

    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.axvline(0, color='black', lw=0.8, ls='--')

    plt.grid(True, alpha=0.1)
    plt.legend(loc='upper left')

    plt.savefig(f'{POSES_DIR}/Poses_Combined_[LessMore,Original]_Comparisons.png', dpi=150, bbox_inches='tight')
    plt.close()

def Plot_Poses_Individuals(poses_combined, title):
            plt.figure(figsize=(10, 10))
            plt.scatter(poses_combined[:, 0], poses_combined[:, 1], alpha=0.3, s=5)
            plt.xlabel('Pose X — With Displaced ')
            plt.ylabel('Pose X — Without Displaced')
            plt.title('Displaced vs Original Pose')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{POSES_DIR}/poses_comparison_{title}.png', dpi=150, bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(8, 8)) # Mudado para quadrado (8x8) para manter a proporção dos eixos


if __name__ == '__main__':
    args = Get_Args()

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec 
    CAP_GEN = args.cap_gen 
    DATASET = args.dataset
    
    padding_mode_sift = 'border' if 'CIFAR' in DATASET else 'zeros'

    POSES_DIR = 'Poses'
    POSES_DIR_Images = f'{POSES_DIR}/Images'

    os.makedirs(POSES_DIR_Images, exist_ok=True)

    dataset_class = getattr(datasets, DATASET)
    trainset = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())

    testeloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample = testeloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN)

    capL_test.load_state_dict(torch.load("Results/MNIST/64_25_40_40_0.001/best_model.pth", map_location=DEVICE))

    capL_test.eval()
    len_batch_size = len(testeloader) - 2 # To save last Input, Output, Target images of each epoch
    study_capsule_index = 2 # Capsule index to study (0 to 24)
    poses_y_original = []
    all_y_poses_less = []
    all_y_poses_more = []
    poses_x_original = []
    all_x_poses_less = []
    all_x_poses_more = []
    with torch.no_grad():
        for i, (img, _) in enumerate(testeloader):
            img = img.to(DEVICE)
            dxy_zeros = torch.zeros(img.size(0), 2, device=DEVICE)
            img_cap_dxyzeros, poses_delzeros, poses_prob = capL_test(img, dxy_zeros, sep=True)
            # print(poses_delzeros[study_capsule_index].size()) # torch.Size([25, 64, 2])
            poses_x_original.append(poses_delzeros[study_capsule_index, :, 0].cpu().numpy()) # Save original x poses for all batches
            poses_y_original.append(poses_delzeros[study_capsule_index, :, 1].cpu().numpy()) # Save original y poses for all batches

            # -3 ou +3 shift doesn't mean it's a 3 pixel shift
            target_less, dxy_less = BatchShift(img, -3, padding_mode_sift, DEVICE)
            target_more, dxy_more = BatchShift(img, 3, padding_mode_sift, DEVICE)

            out_less, poses_less, poses_prob_less = capL_test(target_less, dxy_less, sep=True)
            out_more, poses_more, poses_prob_more = capL_test(target_more, dxy_more, sep=True)

            all_x_poses_less.append(poses_less[study_capsule_index, :, 0].cpu().numpy()) 
            all_y_poses_less.append(poses_less[study_capsule_index, :, 1].cpu().numpy()) 

            all_x_poses_more.append(poses_more[study_capsule_index, :, 0].cpu().numpy()) 
            all_y_poses_more.append(poses_more[study_capsule_index, :, 1].cpu().numpy()) 

            # img_cap_dxyzeros = img_cap_dxyzeros.view(-1, IMG_C, IMG_H, IMG_W)
            # out_less = out_less.view(-1, IMG_C, IMG_H, IMG_W)
            # out_more = out_more.view(-1, IMG_C, IMG_H, IMG_W)
            # #img_cap_dxyzeros = torch.sigmoid(img_cap_dxyzeros).detach().cpu() if 'CIFAR' not in DATASET else img_cap_dxyzeros.clamp(0, 1).detach().cpu()
            # out_less = torch.sigmoid(out_less).detach().cpu() if 'CIFAR' not in DATASET else out_less.clamp(0, 1).detach().cpu()
            # out_more = torch.sigmoid(out_more).detach().cpu() if 'CIFAR' not in DATASET else out_more.clamp(0, 1).detach().cpu()
            # if (i == 0 or i == len_batch_size): 
            #     Save_In_Out_Target_Images(img, target_less, target_more, i, POSES_DIR_Images, 'function_of_shift')
            #     Save_In_Out_Target_Images(img, out_less, out_more, i, POSES_DIR_Images, 'capsule_output')
        
        # All poses for the "img"
        poses_x_original = np.concatenate(poses_x_original, axis=0)  # (10000,) 
        poses_y_original = np.concatenate(poses_y_original, axis=0)  # (10000,) 

        # All poses for the "target_less" images with -3 shift
        all_x_poses_less = np.concatenate(all_x_poses_less, axis=0)  # (10000,) 
        all_y_poses_less = np.concatenate(all_y_poses_less, axis=0)  # (10000,) 
        
        # All poses for the "target_more" images with +3 shift
        all_x_poses_more = np.concatenate(all_x_poses_more, axis=0)  # (10000,)
        all_y_poses_more = np.concatenate(all_y_poses_more, axis=0)  # (10000,)

        poses_combined_less = np.stack([all_x_poses_less, poses_x_original], axis=1)
        poses_combined_more = np.stack([all_x_poses_more, poses_x_original], axis=1)

        Save_Plot_Poses_LessOriginal_MoreOriginal(poses_combined_less, poses_combined_more, POSES_DIR)
        Plot_Poses_Individuals(poses_combined_less, 'LessOriginal')
        Plot_Poses_Individuals(poses_combined_more, 'MoreOriginal')
        












