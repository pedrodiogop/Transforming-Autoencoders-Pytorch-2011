import torch.nn.functional as F
import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

def BatchShift_torch(imbatch: torch.Tensor, dx, padding_mode_sift, device, pose_dim):

    B, _, H, W = imbatch.shape
    
    R = torch.zeros(B, pose_dim, device=device)
    
    #R[:,0] = 2

    #dx_norm = R[:, 0] / (W / 2.0)
    # dy_norm = R[:, 1] / (W / 2.0)

    theta = torch.zeros(B, 2, 3, device=device)


    theta[:, 0, 0] = 1.0     
    theta[:, 1, 1] = 1.0     
    theta[:,0,1] = 0.0
    theta[:,1,0] = 0
    theta[:, 0, 2] = 0     # translação X
    theta[:, 1, 2] = 0    # translação Y

    grid = F.affine_grid(theta, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear', 
                             padding_mode=padding_mode_sift, align_corners=False)

    # angle_deg = (
    #     torch.rand(B, device=device)
    #     * (angle_range[1] - angle_range[0])
    #     + angle_range[0]
    # )

    angle_deg = torch.tensor(0.0)
    # angle_deg = torch.tensor(10.0)
    theta = angle_deg * torch.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # rotação 0: 
    # cos_theta = 1
    # sin_theta = 0

    dx_norm = 0 / (W / 2.0)
    dy_norm = 0 / (W / 2.0)

    R[:,0] = dx_norm # 0.214 -> 3 pixeis 
    R[:,1] = dy_norm # 0.214 -> 3 pixeis 
    R[:,2] = cos_theta
    R[:,3] = sin_theta
    R[:,4] = 1 # [0.9, 1.5]
    R[:,5] = 0.0 # [-30, 30]
    R[:,6] = 0.0

    return shifted, R

def Save_In_Out_Target_Images(inp, target_right, target_left, img_cap_dxyzeros, out_right, out_left, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES):
    inp = inp.detach().cpu()
    target_right = target_right.detach().cpu()
    target_left = target_left.detach().cpu()
    img_cap_dxyzeros = torch.sigmoid(img_cap_dxyzeros).detach().cpu()
    out_right = torch.sigmoid(out_right).detach().cpu()
    out_left =  torch.sigmoid(out_left).detach().cpu()
    

    batch = torch.cat([inp, target_right, target_left, img_cap_dxyzeros, out_right, out_left], dim=3)
    # batch1 = torch.cat([img_cap_dxyzeros, out_right, out_left], dim=3) 
    # batch2 = torch.cat([batch, batch1], dim=3) 

    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5)
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