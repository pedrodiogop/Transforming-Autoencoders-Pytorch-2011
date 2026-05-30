import torch.nn.functional as F
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

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

def Save_In_Out_Target_Images(inp, target_right, target_left, img_cap_dxyzeros, out_right, out_left, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES):

    batch = torch.cat([inp, target_right, target_left], dim=3)
    batch1 = torch.cat([img_cap_dxyzeros, out_right, out_left], dim=3) 
    batch2 = torch.cat([batch, batch1], dim=2) 
    im_tensor = torchvision.utils.make_grid(batch2, nrow=8, normalize=False, padding=2, pad_value=0.5)
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    

    caminho = os.path.join(RESULTS_DIR_IN_OUT_TARGET_IMAGES, f'original_siftRight_siftLeft_modelOriginal_modelRight_modelLeft_batch_{i:03d}.png')
    plt.imsave(caminho, img)

def Save_Plot_Poses_LessOriginal_MoreOriginal(poses_combined_less, poses_combined_more, POSES_DIR, cap_idx, SIZE_DISPLACEMENT, coordinate):

    plt.figure(figsize=(8, 8))

    # Right shift (+3px)
    x_right = poses_combined_less[:, 0]
    x_original = poses_combined_less[:, 1]
    # Least Squares 
    m_right, b_right = np.polyfit(x_right, x_original, 1)
    y_pred_right = m_right * x_right + b_right
    plt.scatter(x_right, x_original, alpha=0.2, s=5, color='tab:green', label=f'Shift +{SIZE_DISPLACEMENT}px')    
    plt.plot(x_right, y_pred_right, color='darkgreen', lw=2,
        label=f'Linear Fit +{SIZE_DISPLACEMENT}px (slope={m_right:.2f})')
    
    # Left shift (-3px)
    x_left = poses_combined_more[:, 0]
    x_original_v1 = poses_combined_more[:, 1]
    # Least Squares
    m_left, b_left = np.polyfit(x_left, x_original_v1, 1)
    y_pred_left = m_left * x_left + b_left
    plt.scatter(x_left, x_original_v1, alpha=0.2, s=5, color='tab:blue', label=f'Shift -{SIZE_DISPLACEMENT}px')
    plt.plot(x_left, y_pred_left, color='darkblue', lw=2,
        label=f'Linear Fit -{SIZE_DISPLACEMENT}px (slope={m_left:.2f})')
        
    # Plot Configurations
    plt.xlabel('Estimated Pose — Shifted Image')
    plt.ylabel('Estimated Pose — Original Image')
    plt.title(f'Pose Equivariance — Capsule {cap_idx} - Coordinate {coordinate}')

    plt.axhline(0, color='black', lw=0.8, ls='--')
    plt.axvline(0, color='black', lw=0.8, ls='--')

    plt.grid(True, alpha=0.1)
    plt.legend(loc='upper left')

    plt.savefig(f'{POSES_DIR}/Pose_Equivariance_Cap{cap_idx:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()