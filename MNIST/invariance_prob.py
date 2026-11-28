#   Transformação T1 (aplicada a original image) Original Image -> T1 -> Input Image 
#   Horizontal and vertical translation (pixels) [−2, 2]; 
#   Rotation (degrees) [−15, 15], 
#   Scale [0.95, 1.2]; 
#   Horizontal and vertical shear (degrees) [−10, 10]
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args, set_seed
import torch
from CapLayer import CapLayer
import torch.nn as nn
import os
import torchvision
import matplotlib.pyplot as plt
import pandas as pd

def _sample_T_R(B, H, W, dxdy, angle, scale, shear, device, pose_dim):

    tx = dxdy[0]
    ty = dxdy[1]
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    theta = torch.tensor(angle * torch.pi / 180)
    #theta = angle * torch.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    shear_x = torch.tan(shear[0] * torch.pi / 180)
    shear_y = torch.tan(shear[1] * torch.pi / 180)

    T = torch.zeros(B, 2, 3, device=device)
    T[:, 0, 0] = cos_theta * scale
    T[:, 0, 1] = -sin_theta * scale + shear_x
    T[:, 1, 0] = sin_theta * scale + shear_y
    T[:, 1, 1] = cos_theta * scale
    T[:, 0, 2] = dx_norm
    T[:, 1, 2] = dy_norm

    return T


def _warp(img, T, padding_mode):
    grid = F.affine_grid(T, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=False)

# def BatchShift(imbatch: torch.Tensor,
#                               dxdy, angle_range, scale_range, shear_range,           # imbatch    -> inp_image
#                               padding_mode_sift, device, pose_dim,
#                               dxdy_i = [-2, 2], angle_range_i = [-15, 15], scale_range_i = [0.95, 1.2], shear_range_i = [-10, 10],   # inp_image  -> target
#                             ):

def BatchShift(imbatch: torch.Tensor, padding_mode_sift, device, pose_dim, t_input):

    B, _, H, W = imbatch.shape

    # imbatch -> inp_image
    T_in = _sample_T_R(B, H, W, [t_input[0],t_input[1]], t_input[2], t_input[3], [t_input[4],t_input[5]],
                       device, pose_dim)
    inp_image = _warp(imbatch, T_in, padding_mode_sift)

    # inp_image -> target 
    # _, R = _sample_T_R(B, H, W, dxdy, angle_range, scale_range, shear_range,
    #                          device, pose_dim)
    # target = _warp(inp_image, T, padding_mode_sift)

    return inp_image

def prob_media(all_prob_caps, vector_classes, n_classes=10):
    n_caps = all_prob_caps.shape[0]
    prob_soma = np.zeros((n_caps, n_classes), dtype=float)   # [40, 10]
    prob_std = np.zeros((n_caps, n_classes), dtype=float)    # [40, 10]
    for k in range(n_caps):
        for c in range(n_classes):
            p_kc = all_prob_caps[k, vector_classes == c]
            prob_soma[k, c] = p_kc.sum().item()
            prob_std[k, c] = p_kc.std().item()
    n_por_classe = np.array([(vector_classes == c).sum().item() for c in range(n_classes)])
    prob_media = prob_soma / n_por_classe
    return prob_soma, prob_media, prob_std, n_por_classe


def probs_com_transf(model, loader, transf_input):
    """p de cada cápsula para imagens transformadas (distribuição de treino)."""
    probs_list, labels_list = [], []
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(DEVICE)
            inp_img = BatchShift(
                img, padding_mode_sift, DEVICE, LEN_POSE, transf_input)
            _, prob = model(inp_img, [0, 0, 0, 1, 0, 0], sep=True)
            probs_list.append(prob.squeeze(-1).cpu())     # [40, B]
            # labels_list.append(labels.cpu())
    # print(len(probs_list))
    all_prob_caps = torch.cat(probs_list, dim=1)                       # [40, N_teste]
    # classes = torch.cat(labels_list, dim=0)                      # [N_teste]
    return all_prob_caps

def probs_sem_transf(model, loader):
    """p de cada cápsula para as imagens originais (transformação identidade)."""
    identidade = torch.tensor([0., 0., 0., 1., 0., 0.], device=DEVICE)  # [tx, ty, angle, scale, shear_x, shear_y]
    probs_list, labels_list = [], []
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(DEVICE)
            dxy_id = identidade.repeat(img.size(0), 1)     # [B, 6], uma por imagem
            _, prob = model(img, dxy_id, sep=True)
            probs_list.append(prob.squeeze(-1).cpu())
    all_prob_caps_without = torch.cat(probs_list, dim=1)
    return all_prob_caps_without

if __name__ == '__main__':
    
    args = Get_Args()

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
    # SEED = args.seed
    SEED = 42
    SCALE_MIN = args.scale_min
    SCALE_MAX = args.scale_max
    SHEAR = args.shear
    lr = args.lr
    set_seed(SEED)
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{SCALE_MIN}_{SCALE_MAX}_{SHEAR}_{lr}_{42}_20'
    RESULTS_DIR_INVARIANCE_PROB = f'{RESULTS_DIR}/Invariance_Prob'
    os.makedirs(RESULTS_DIR_INVARIANCE_PROB, exist_ok=True)

    padding_mode_sift = 'zeros'    
    dataset_class = getattr(datasets, DATASET)
    test_set = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
        
    testeloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    sample = testeloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    # crit = nn.BCEWithLogitsLoss() 

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()

    TX = [[-3, 0, 0, 1, 0, 0], [-2, 0, 0, 1, 0, 0], [-1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [2, 0, 0, 1, 0, 0], [3, 0, 0, 1, 0, 0]]
    # TY = [[0 , -3, 0, 1, 0, 0], [0, -2, 0, 1, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 2, 0, 1, 0, 0], [0, 3, 0, 1, 0, 0]]
    # ROTACAO = [[0 , 0, -20, 1, 0, 0], [0, 0, -15, 1, 0, 0], [0, 0, -10, 1, 0, 0], [0, 0, -5, 1, 0, 0], [0, 0, 0, 1, 0, 0], [0 , 0, 5, 1, 0, 0], [0, 0, 10, 1, 0, 0], [0, 0, 15, 1, 0, 0], [0, 0, 20, 1, 0, 0]]
    # SCALE = [[0 , 0, 0, 1.3, 0, 0], [0, 0, 0, 1.2, 0, 0], [0, 0, 0, 1.1, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0.97, 0, 0], [0 , 0, 0, 0.95, 0, 0], [0, 0, 0, 0.9, 0, 0]]
    # SX = [[0 , 0, 0, 1, -15, 0], [0, 0, 0, 1, -10, 0], [0, 0, 0, 1, -5, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 5, 0], [0, 0, 0, 1, 10, 0], [0 , 0, 0, 1, 15, 0]]
    # SY = [[0 , 0, 0, 1, 0, -15], [0, 0, 0, 1, 0, -10], [0, 0, 0, 1, 0, -5], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 5], [0, 0, 0, 1, 0, 10], [0 , 0, 0, 1, 0, 15]]
    
    all_prob_caps_without = probs_sem_transf(capL_test, testeloader)
    for transf in TX:
        all_prob_caps = probs_com_transf(capL_test, testeloader, transf)
        # all_prob_caps -> [num_caps, N_teste_images]
        transf_caps_all_prob.append(all_prob_caps)

    print(len(transf_caps_all_prob))
    exit()
    _, matrix_prob_media, matrix_prob_std, _ = prob_media(all_prob_caps, vector_classes)
    _, matrix_prob_media_without, matrix_prob_std_without, _ = prob_media(all_prob_caps_without, vector_classes)

    media_por_capsula = all_prob_caps.mean(dim=1).numpy()                       # [40] média sobre todas as imagens
    std_por_capsula = all_prob_caps.std(dim=1).numpy()

    media_por_capsula_without = all_prob_caps_without.mean(dim=1).numpy()       # [40]  
    std_por_capsula_without = all_prob_caps_without.std(dim=1).numpy()