import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid, save_image
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args, set_seed, BatchShift_torch_Rotation
# from Class_HeadPose import CustomImageDataset
import torch
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torchvision
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import torch.nn.functional as F

def _sample_T_R(B, H, W, dxdy, angle_range, scale_range, shear_range, device, pose_dim, shifts_px):
    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    #tx = torch.randint(low=dxdy[0], high=dxdy[1]+1, size=(B,), device=device).float()
    #ty = torch.randint(low=dxdy[0], high=dxdy[1]+1, size=(B,), device=device).float()
    shifts = torch.tensor(shifts_px, dtype=torch.float32, device=device)

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    # dx_norm = shifts / (W / 2.0)
    dx_norm = 0
    dy_norm = 0
    # dy_norm = ty / (H / 2.0)
    #dy_norm = shifts / (W / 2.0)

    angle_deg = (
        torch.rand(B, device=device)
        * (angle_range[1] - angle_range[0])
        + angle_range[0]
    )
    # angle_deg = shifts
    angle_normalized = angle_deg / 180.0
    theta = angle_deg * torch.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    scale = torch.rand(B, device=device) * (scale_range[0] - scale_range[1]) + scale_range[1]
    #scale = shifts
    shear_angle_deg_x = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_angle_deg_y = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_angle_deg_y = shifts
    shear_x = torch.tan(shear_angle_deg_x * torch.pi / 180)
    shear_y = torch.tan(shear_angle_deg_y * torch.pi / 180)

    # ── 3. Vetor de pose ──────────────────────────────────────────────────────
    R = torch.zeros(B, pose_dim, device=device)
    R[:, 0] = - dx_norm
    R[:, 1] = dy_norm
    R[:, 2] = angle_normalized
    R[:, 3] = 1.0 - scale
    R[:, 4] = shear_x
    R[:, 5] = shear_y

    # ── 4. Matriz T ───────────────────────────────────────────────────────────
    T = torch.zeros(B, 2, 3, device=device)
    T[:, 0, 0] = cos_theta * scale
    T[:, 0, 1] = -sin_theta * scale + shear_x
    T[:, 1, 0] = sin_theta * scale + shear_y
    T[:, 1, 1] = cos_theta * scale
    T[:, 0, 2] = dx_norm
    T[:, 1, 2] = dy_norm

    return T, R


def _warp(img, T, padding_mode):
    grid = F.affine_grid(T, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=False)


# def BatchShift_torch_Rotation(imbatch: torch.Tensor,
#                               dxdy = [-5, 5], angle_range=[-45, 45], scale_range=[0.85, 1.5], shear_range=[-25, 25],           # imbatch    -> inp_image
#                               padding_mode_sift= 'zeros', device='cuda', pose_dim = 6,
#                               dxdy_i = [-2, 2], angle_range_i = [-15, 15], scale_range_i = [0.95, 1.2], shear_range_i = [-10, 10],   # inp_image  -> target
#                             ):
def BatchShift_torch_Rotation(imbatch: torch.Tensor, shifts_px,
                              dxdy = [-0, 0], angle_range=[0, 0], scale_range=[1, 1], shear_range=[0, 0],           # imbatch    -> inp_image
                              padding_mode_sift= 'zeros', device='cuda', pose_dim = 6,
                              dxdy_i = [-2, 2], angle_range_i = [-15, 15], scale_range_i = [0.95, 1.2], shear_range_i = [-10, 10],   # inp_image  -> target
                            ):
    #B, _, H, W = imbatch.shape
    imbatch = imbatch.repeat(len(shifts_px), 1, 1, 1)
    B, _, H, W = imbatch.shape
    # imbatch -> inp_image
    # T_in, _ = _sample_T_R(B, H, W, dxdy_i, angle_range_i, scale_range_i, shear_range_i,
    #                    device, pose_dim)
    # inp_image = _warp(imbatch, T_in, padding_mode_sift)

    # inp_image -> target 
    T, R = _sample_T_R(B, H, W, dxdy, angle_range, scale_range, shear_range,
                             device, pose_dim, shifts_px)
    target = _warp(imbatch, T, padding_mode_sift)

    return target, R


#RANDOM_TRANSLATION = 0.0 #7
#ROTATION_ANGLE = 30.0  #180
# SCALE = 1.5 # 0.9
# SHEAR = 0.3 # -0.3

if __name__ == '__main__':
    
    args = Get_Args()

    # The architecture and hyperparameters must be the same as those used in training; 
    # The folder name where the model is stored contains all the necessary data.
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec # encode the image
    CAP_GEN = args.cap_gen # decode the image
    DATASET = args.dataset
    LEN_POSE = args.len_pose
    RANDOM_TRANSLATION = args.random_translation
    ROTATION_ANGLE = args.rotation_angle
    SEED = args.seed
    CUSTOM_DATASET = args.custom_dataset
    SEED = args.seed
    SCALE_MIN = args.scale_min
    SCALE_MAX = args.scale_max
    SHEAR = args.shear
    
    lr = args.lr
    set_seed(SEED)
    
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{SCALE_MIN}_{SCALE_MAX}_{SHEAR}_{lr}_{SEED}_20'
    RESULTS_DIR_LATENT_TRASVERSAL = f'{RESULTS_DIR}/Latent_Trasversal'

    os.makedirs(RESULTS_DIR_LATENT_TRASVERSAL, exist_ok=True)

    padding_mode_sift = 'zeros'    

    test_set = datasets.MNIST(root="tmp", train=False, download=True, transform=ToTensor())
    # index translation x = 10, 7
    # index translation y= 11, 14
    # index rotation = 0, 1
    # index scale = 3, 4
    # index shear x= 14, 15
    # index shear y= 17, 230
    image, _ = test_set[230]
    IN_DIM = image.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = image.shape
    image = image.unsqueeze(0)
    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()
    with torch.no_grad():
        image = image.to(DEVICE)
        #target, dxy = BatchShift_torch_Rotation(image, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], 'zeros', 'cpu', 7)
        # shifts_px = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10] Tx & Ty
        # shifts_px = [80, 60, 50, 40, 20, 0, -20, -40, -50, -60, -80] R
        # shifts_px = [2, 1.8, 1.6, 1.4, 1.2, 1, 0.9, 0.85, 0.8, 0.65, 0.45] # S
        # shifts_px = [-60, -48, -36, -24, -12, 0, 12, 24, 36, 48, 60] # Shear
        
        shifts_px = [-60, -48, -36, -24, -12, 0, 12, 24, 36, 48, 60] # S
        target, dxy = BatchShift_torch_Rotation(image, shifts_px)
        inp_batch = image.repeat(len(shifts_px), 1, 1, 1)

        out = capL_test(inp_batch, dxy)
        out = out.view(-1, IMG_C, IMG_H, IMG_W)
        out = torch.sigmoid(out)
        grid = make_grid(torch.cat([target, out], dim=0),
                         nrow=len(shifts_px), padding=1, pad_value=1)
        save_image(grid, f'{RESULTS_DIR_LATENT_TRASVERSAL}/latent_transversal.png')


        # grid = make_grid(torch.cat([image, target], dim=3), nrow=1, padding=2, pad_value=1)
        # save_image(grid, f'{RESULTS_DIR_LATENT_TRASVERSAL}/latent_transversal.png')
    # grid = make_grid(
    #     torch.cat([image01, image03, image05, image07, image02, image, image10, image08, image09, image04], dim=2),
    #     nrow=1 
    #     )   
    
    # save_image(grid, "Transformation/mnist_comparison.png")