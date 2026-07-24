import os
import torch
import torchvision
import matplotlib.pyplot as plt
from aux_functions import Get_Args
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Feature_Map_CapLayer import Feature_Map_CapLayer
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_pil_image

# def Save_Feature_Map_Images(inp, target, out, caps_imgs, label, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET, transformations):
#     transformations = '_'.join(str(round(float(v), 2)) for v in transformations.flatten())
#     file_path = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}/Capsules/{label}/{transformations}'
#     os.makedirs(file_path, exist_ok=True) # save input, output and target images for each epoch
#     inp = inp.detach().cpu()
#     vmin = out.min()
#     vmax = out.max()
#     print(vmin.item(), vmax.item())
#     out = torch.sigmoid(out).detach().cpu() if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()
#     vmin = out.min()
#     vmax = out.max()
#     print(vmin.item(), vmax.item())

#     if target is not False and target is not None:
#         target = target.detach().cpu()
#     #    batch = torch.cat([inp, target, out], dim=3)
#     # else: 
#     #    batch = torch.cat([inp, out], dim=3)

#     batch = torch.stack([inp.squeeze(0), target.squeeze(0), out.squeeze(0)], dim=0)
#     save_image(batch, f'{file_path}/Reconstruction.png', nrow=3, padding=1, normalize=False, pad_value=0.5)

#     # print(caps_imgs.size()) # torch.Size([25, 1, 28, 28])
#     save_image(caps_imgs, f'{file_path}/Caps_FeatureMaps.png', nrow=1, padding=1, normalize=False, pad_value=0.5)
#     # grid = grid.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8) What happen in the background when using save_image.

#     two_caps_images = caps_imgs[[0,1]]
#     sum_caps = torch.sum(two_caps_images, dim=0)
#     sum_caps_expandido = sum_caps.unsqueeze(0)
#     sum_caps_image = torch.cat([two_caps_images, sum_caps_expandido], dim=0)

#     save_image(sum_caps_image, f'{file_path}/Sum_Caps_FeatureMaps0.png', nrow=2, padding=1, normalize=False, pad_value=0.5)

# angle_range is the range of rotation angles in degrees, e.g., [-30, 30]
def BatchShift_torch(imbatch: torch.Tensor, dxdy, angle_range, padding_mode_sift, device, pose_dim):
    B, _, H, W = imbatch.shape

    # pose_dim deve ser 6 — os 6 elementos da matriz 2×3
    R = torch.zeros(B, pose_dim, device=device, dtype=torch.float32)

    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    dx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
    dy = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()

    angle_deg = (torch.rand(B, device=device)
                 * (angle_range[1] - angle_range[0])
                 + angle_range[0])
    theta_rad = angle_deg * (torch.pi / 180.0)

    cos_t = torch.cos(theta_rad)
    sin_t = torch.sin(theta_rad)

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = dx / (W / 2.0)
    dy_norm = dy / (H / 2.0)

    # ── 3. Preencher R com os 6 elementos da matriz 2×3 ──────────────────────
    #   | cos(θ)  -sin(θ)   tx |  →  índices [0] [1] [2]
    #   | sin(θ)   cos(θ)   ty |  →  índices [3] [4] [5]
    R[:, 0] = cos_t
    R[:, 1] = -sin_t
    R[:, 2] = dx_norm
    R[:, 3] = sin_t
    R[:, 4] = cos_t
    R[:, 5] = dy_norm

    # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
    T = R.view(B, 2, 3)

    grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear',
                            padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R

# def build_capsule_sum_pyramid_by_max(caps_imgs):
#     """
#     Constrói a pirâmide de somas. Cada nível é ordenado pelo valor máximo
#     de pixel de cada imagem (mais positivo primeiro), incluindo o nível 0
#     (caps_imgs original). O pareamento é sempre feito na ordem já ordenada.
#     """
#     def sort_by_max(tensor):
#         n = tensor.shape[0]
#         maxs = tensor.view(n, -1).max(dim=1).values
#         sort_idx = torch.argsort(maxs, descending=True)  # mais positivo primeiro
#         return tensor[sort_idx], sort_idx

#     current, sort_idx = sort_by_max(caps_imgs)   # nível 0 já ordenado
#     levels = [current]
#     indices = [sort_idx]

#     while current.shape[0] > 1:
#         n = current.shape[0]
#         n_pairs = n // 2

#         even = current[0 : n_pairs * 2 : 2]
#         odd  = current[1 : n_pairs * 2 : 2]
#         summed = even + odd

#         if n_pairs * 2 < n:
#             leftover = current[n_pairs * 2 :]
#             leftover_sum = torch.sum(leftover, dim=0)
#             summed[-1] = summed[-1] + leftover_sum

#         summed_sorted, sort_idx  = sort_by_max(summed)
#         levels.append(summed_sorted)
#         indices.append(sort_idx)
#         current = summed_sorted

#     return levels, indices

def build_capsule_sum_pyramid(caps_imgs):
    # print(caps_imgs.size()) # torch.Size([25, 1, 28, 28])
    """
    Constrói a pirâmide de somas: nível 0 = cápsulas originais,
    nível seguinte = soma de pares consecutivos, e assim sucessivamente
    até restar uma única imagem (a soma de todas as cápsulas).
    """
    levels = [caps_imgs] # convert to list, each dimension is a level of the pyramid
    current = caps_imgs
    # print(len(levels[0][0][0]))

    while current.shape[0] > 1:
        n = current.shape[0] # 25 | 
        n_pairs = n // 2

        # obj[Start : end : setp]
        even = current[0 : n_pairs * 2 : 2]
        odd  = current[1 : n_pairs * 2 : 2]
        summed = even + odd

        # sobra ímpar? absorve na última soma deste nível
        if n_pairs * 2 < n:
            leftover = current[n_pairs * 2 :]
            leftover.squeeze(0)
            summed[-1] = summed[-1] + leftover

        levels.append(summed) 
        current = summed

    return levels

def plot_feature_maps_rows(caps_imgs, inp, out, target, all_probs, file_path, cmap='viridis'):
    # print(caps_imgs.size()) # torch.Size([num_caps, 1, 28, 28])
    rows = build_capsule_sum_pyramid(caps_imgs)
    # print(f"Number of levels in the pyramid: {len(rows)}")
    all_probs = all_probs.detach().cpu().flatten()
    # probs = all_probs[indices[0]]

    nrows = len(rows) * 2          # cada nível ocupa 2 linhas: normalizada + clamped
    ncols_max = caps_imgs.shape[0]

    # posição de coluna: no nível L, cada imagem i fica na coluna i * 2**L
    col_positions = [
        [i * (2 ** level) for i in range(row.shape[0])]
        for level, row in enumerate(rows)
    ]

    fig, axes = plt.subplots(nrows, ncols_max, figsize=(ncols_max * 4, nrows * 4))
    axes = np.array(axes).reshape(nrows, ncols_max)

    for r in range(nrows):
        for c in range(ncols_max):
            axes[r, c].axis('off')

    for level, row_tensor in enumerate(rows):
        imgs = row_tensor.detach().cpu().numpy()
        if imgs.ndim == 4:
            imgs = imgs.squeeze(1)

        positions = col_positions[level]

        row_normal = level * 2
        row_clamped = level * 2 + 1

        for i, col in enumerate(positions):
            img = imgs[i]
            vmin, vmax = img.min(), img.max()

            # linha normal (escala real do valor)
            ax_normal = axes[row_normal, col]
            im = ax_normal.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

            if level == 0:
                ax_normal.set_title(f"Cap {i}; P={all_probs[i]:.3f}\n[{vmin:.2f}, {vmax:.2f}]", fontsize=14)
            else:
                ax_normal.set_title(f'[{vmin:.2f}, {vmax:.2f}]', fontsize=8)

            fig.colorbar(im, ax=ax_normal, fraction=0.046, pad=0.04)
            ax_normal.axis('on')

            # linha clamped (recortada para [0,1], cinzento)
            ax_clamped = axes[row_clamped, col]
            img_clamped = np.clip(img, 0.0, 1.0)
            im_clamped = ax_clamped.imshow(img_clamped, cmap='gray', vmin=0.0, vmax=1.0)
            # ax_clamped.set_title('clamped [0,1]', fontsize=7)
            fig.colorbar(im_clamped, ax=ax_clamped, fraction=0.046, pad=0.04)
            ax_clamped.axis('on')

    plt.tight_layout()

    os.makedirs(file_path, exist_ok=True)

    plt.savefig(f'{file_path}/Sum_Caps_FeatureMaps.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def Input_Target_Output_Images(inp, target, out, file_path):
    inp = inp.detach().cpu()
    out = torch.sigmoid(out).detach().cpu() if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()

    if target is not False and target is not None:
        target = target.detach().cpu()
    # print(f"Input shape: {inp.shape}, Target shape: {target.shape}, Output shape: {out.shape}")

    #final_sum_image = rows[-1].detach().cpu().squeeze(0)  # [C, H, W]
    #final_sum_image = torch.sigmoid(final_sum_image) if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else final_sum_image.clamp(0, 1)

    # batch = torch.stack([inp.squeeze(0), target.squeeze(0), out.squeeze(0), final_sum_image], dim=0)
    batch = torch.stack([inp.squeeze(0), target.squeeze(0), out.squeeze(0)], dim=0)
    # print(f"Batch shape for saving: {batch.shape}")  # [3, C, H, W]
    os.makedirs(file_path, exist_ok=True)
    save_image(batch, f'{file_path}/Reconstruction.png', nrow=4, padding=1, normalize=False, pad_value=0.5)


def PlotGenerativeFeatureMaps(capL, X, transformation, img_h, img_w, file_path, num_capsule, num_generative):

    fig_w = num_generative * 3
    fig_h = num_capsule * 3

    fig, axes = plt.subplots(num_capsule,
                                num_generative,
                                figsize=(fig_w, fig_h))

    # fig.suptitle(
    #     f'Feature maps Generative Units',
    #     fontsize=6
    # )

    with torch.no_grad():

        for k in range(num_capsule):

            cap = capL.caps[k]
            _, _, prob, gen = cap(X, transformation)
            prob_value = prob.mean().item()
            # print(gen.size()) torch.Size([1, 40])

            for p in range(num_generative):

                ax = axes[k][p]

                single_gen = torch.zeros_like(gen)
                single_gen[:, p] = gen[:, p]
                feature = cap.gen_out(single_gen)

                # média no batch
                feature = feature.mean(dim=0) # Se batch for maior que 1 faz a media das imagens caso contrario apenas remove uma dimensao

                feature = feature.view(img_h,img_w)
                feature_clamped = feature.clamp(min=0, max=1)
                ax.imshow(feature_clamped.cpu(),
                        cmap='gray')

                ax.axis('on')

                if k == 0:
                    ax.set_title(
                        f'Gen {p}',
                        fontsize=15
                    )

                if p == 0:  
                    ax.set_ylabel(f'Cap {k}\nProb: {prob_value:.2f}', fontsize=15)

    # plt.subplots_adjust(
    #     wspace=0.10,
    #     hspace=0.10
    # )

    plt.tight_layout()


    os.makedirs(file_path, exist_ok=True)
    plt.savefig(
        f'{file_path}/FeatureMaps_Generetive_Units.png',
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)



if __name__ == '__main__':
    args = Get_Args()
    # If you want to see your GPU or CPU in action, you can use the following code to check if PyTorch recognizes it and to set the device accordingly:
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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
    
    lr = args.lr

    # RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{lr}_{SEED}'
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{lr}'
    RESULTS_DIR_FEATURE_MAP = f'{RESULTS_DIR}/Feature_Map'

    if 'CIFAR' in DATASET:
        padding_mode_sift = 'reflection' if DEVICE == 'mps' else 'border'
    else:
        padding_mode_sift = 'zeros'

    dataset_class = getattr(datasets, DATASET)
    test_set = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
    testeloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    image_index = 17  # Index of the image to visualize
    # testeloader.dataset[i][j] -> (image,label)
    sample = testeloader.dataset[image_index][0]  # (C, H, W)
    label = testeloader.dataset[image_index][1]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_feature_map = Feature_Map_CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_feature_map = capL_feature_map.to(DEVICE)
    # crit = nn.MSELoss() if 'CIFAR' in DATASET else nn.BCEWithLogitsLoss() 

    capL_feature_map.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_feature_map.eval()
    
    with torch.no_grad():
        sample = sample.unsqueeze(0) # Add a batch dimension to the sample
        sample = sample.to(DEVICE)
        target, transformations = BatchShift_torch(sample, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], padding_mode_sift, DEVICE, LEN_POSE)
        RANDOM_TRANSFORMATIONS = f'{RANDOM_TRANSLATION}_{ROTATION_ANGLE}'

        capsule_stacked, reconstruction_image, all_probs, _ = capL_feature_map(sample, transformations)
        # print(capsule_stacked.size()) torch.Size([25, 1, 784])

        reconstruction_image = reconstruction_image.view(-1, IMG_C, IMG_H, IMG_W)
        num_caps = capsule_stacked.shape[0]
        caps_imgs = capsule_stacked.view(num_caps, IMG_C, IMG_H, IMG_W)
        # print(caps_imgs.size()) # torch.Size([25, 1, 28, 28])

        transformations_str = '_'.join(str(round(float(v), 2)) for v in transformations.flatten())
        file_path = f'{RESULTS_DIR_FEATURE_MAP}/Capsules/{label}/{transformations_str}'



        plot_feature_maps_rows(caps_imgs, sample, reconstruction_image, target, all_probs, file_path)
        Input_Target_Output_Images(sample, target, reconstruction_image, file_path)
        PlotGenerativeFeatureMaps(capL_feature_map, sample, transformations, IMG_H, IMG_W, file_path, NUM_CAPS, CAP_GEN)

       
       