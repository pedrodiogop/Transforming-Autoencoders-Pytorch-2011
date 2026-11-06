import argparse
import random
# from dateutil import parser
from matplotlib import lines
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
# Code in poses.py
# def Plot_Poses(poses, RESULTS_DIR_POSES, epoch):
#     poses = poses.detach().cpu().numpy()

#     x = poses[:, 0]
#     y = poses[:, 1]

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.scatter(x, y, alpha=0.6, edgecolors='w', label='Poses Médias (R)')

#     ax.axhline(0, color='black', lw=1, ls='--') 
#     ax.axvline(0, color='black', lw=1, ls='--') 

#     ax.set_title("Distribuição das Poses Médias no Batch")
#     ax.set_xlabel("Coordenada X")
#     ax.set_ylabel("Coordenada Y")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     plt.savefig(f'{RESULTS_DIR_POSES}/Poses_Ep_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
#     plt.close(fig)

def Get_Args():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='cuda',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=64,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=400,     help='Number of epochs to train')
    parser.add_argument('--num_caps',   type=int,   default=25,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=40,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=40,   help='Capsule generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='MNIST', )
    parser.add_argument('--len_pose',    type=int,   default=7, help='Capsule pose vector length. Minimum need to be 4, this includes translation and rotation. Use 2 for strict spatial equivariance analysis, or > 2 to prioritize image reconstruction capacity.')
    parser.add_argument('--random_translation',    type=int,   default=5, help='To control the size of the displacement, if want to train just for reconstruction set this to 0')
    parser.add_argument('--rotation_angle',    type=int,   default=45, help='To control range of rotation angles.')
    parser.add_argument('--scale_min',    type=float,   default=1.5, help='To control range of scale min.')
    parser.add_argument('--scale_max',    type=float,   default=0.85, help='To control range of scale max.')
    parser.add_argument('--shear',    type=int,   default=25, help='To control range of shear angles.')
    parser.add_argument('--seed',    type=int,   default=42, help='Random seed for reproducibility.')
    parser.add_argument('--custom_dataset', action='store_true', help='Specifically for test.py script. To use this feature, you must create a folder named "Mine_Dataset" inside the folder "Test" and place your custom dataset inside it.')

    return parser.parse_args()


def BatchShift_torch_Rotation(imbatch: torch.Tensor, dxdy, angle_range, scale_range, shear_range, padding_mode_sift, device, pose_dim):
    B, _, H, W = imbatch.shape

    R = torch.zeros(B, pose_dim, device=device)

    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    tx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
    ty = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    # Generate numbers between - angle_range e angle_range
    angle_deg = (
        torch.rand(B, device=device)
        * (angle_range[1] - angle_range[0])
        + angle_range[0]
    )

    angle_normalized = angle_deg / 180.0

    # Convert to radianos, because torch only accept that
    theta = angle_deg * torch.pi / 180

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # 0°   → [ 1,  0]
    # 90°  → [ 0,  1]
    # 180° → [-1,  0]
    # 270° → [ 0, -1]
    # 360° → [ 1,  0]

    # scale = torch.rand(B, device=device) * (1.5 - 0.90) + 0.90 
    scale_x = torch.rand(B, device=device) * (scale_range[0] - scale_range[1]) + scale_range[1]
    scale_y = torch.rand(B, device=device) * (scale_range[0] - scale_range[1]) + scale_range[1]

    # shear_angle_deg_x = torch.rand(B, device=device) * (17 + 17) - 17
    # shear_angle_rad_x = shear_angle_deg_x * torch.pi / 180
    
    # shear_angle_deg_y = torch.rand(B, device=device) * (17 + 17) - 17
    # shear_angle_rad_y = shear_angle_deg_y * torch.pi / 180
    shear_angle_deg_x = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_angle_deg_y = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]

    shear_angle_rad_x = shear_angle_deg_x * torch.pi / 180
    shear_angle_rad_y = shear_angle_deg_y * torch.pi / 180

    shear_x = torch.tan(shear_angle_rad_x)
    shear_y = torch.tan(shear_angle_rad_y)
    

    R[:,0] = - dx_norm
    R[:,1] = dy_norm
    R[:,2] = angle_normalized 
    #R[:,3] = sin_theta
    R[:,3] = 1.0 - scale_x
    R[:,4] = 1.0 - scale_y
    R[:,5] = shear_x
    R[:,6] = shear_y

    # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
    T = torch.zeros(B,2,3,device=device)

    T[:,0,0] = cos_theta * scale_x
    T[:,0,1] = -sin_theta * scale_y + shear_x
    T[:,1,0] = sin_theta * scale_x + shear_y
    T[:,1,1] = cos_theta * scale_y

    T[:,0,2] = dx_norm
    T[:,1,2] = dy_norm

    grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear',
                            padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R

def save_summary_to_file(model_stats, RESULTS_DIR):
    summary_str = str(model_stats)

    with open(os.path.join(RESULTS_DIR, "model_summary.txt"), "w") as f:
        f.write(summary_str)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multiples GP

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def PlotGenrative(epoch, capL, img_c, img_h, img_w, RESULTS_DIR_GENERATIVE, num_capsule, num_generative):
    os.makedirs(RESULTS_DIR_GENERATIVE, exist_ok=True)
    # Tamanho dinâmico baseado na grelha
    fig_w = num_generative * 1.2
    fig_h = num_capsule * 1.2
    fig, axes = plt.subplots(num_capsule, num_generative, figsize=(fig_w, fig_h))
    fig.suptitle(f'Pesos Generativos gen_out — Época {epoch+1}', fontsize=6)
    for k in range(num_capsule): # capsules 
        weights = capL.caps[k].gen_out.weight.data.cpu() 
        for p in range(num_generative):  # Generative diemnsion    
            ax = axes[k][p]
            if img_c == 1:
                ax.imshow(weights[:, p].view(img_h, img_w), cmap='gray')
            else: # Cifar
                w = weights[:, p].view(img_h, img_w, img_c)
                w = (w - w.min()) / (w.max() - w.min())  # normaliza para [0,1]
                ax.imshow(w.numpy())
            ax.axis('off')
            if k == 0:
                ax.set_title(f'Generative {p}', fontsize=4)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # em vez de tight_layout
    plt.savefig(f'{RESULTS_DIR_GENERATIVE}/weight_gen_out_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    
def Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS):
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, label='Loss', linewidth=0.8)
            
    # if len(loss_history) >= window: # The real trend, without the noise.
    #     moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
    #     ax.plot(range(window-1, len(loss_history)), moving_avg, 
    #             label=f'Média móvel ({window})', color='red', linewidth=1.5)
            
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Loss')
    ax.set_title(f'Função de Custo — Época {epoch+1}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{RESULTS_DIR_LOSS}/Loss_Ep_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def Save_In_Out_Target_Images(inp, target, out, epoch, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch
    inp = inp.detach().cpu()
    out = torch.sigmoid(out).detach().cpu() if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()

    if target is not False and target is not None:
        target = target.detach().cpu()
        batch = torch.cat([inp, target, out], dim=3)
    else: 
        batch = torch.cat([inp, out], dim=3)
         
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=False, padding=2, pad_value=0.5)
    # To have the real values we need to set normalize=False. 
    # This way the reconstrution image is not manipulated from the original
    # im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5) 
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    # img = np.clip(img, 0, 1) 
    
    diretorio = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}/Epoch_{epoch:03d}'
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f'batch_{i:05d}.png')
    plt.imsave(caminho, img)


# angle_range is the range of rotation angles in degrees, e.g., [-30, 30]
def BatchShift_torch(imbatch: torch.Tensor, dxdy, angle_range, padding_mode_sift, device, pose_dim):
    B, _, H, W = imbatch.shape

    R = torch.zeros(B, pose_dim, device=device)

    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    tx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
    ty = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    # Generate numbers between - angle_range e angle_range
    angle_deg = (
        torch.rand(B, device=device)
        * (angle_range[1] - angle_range[0])
        + angle_range[0]
    )

    # Convert to radianos, because torch only accept that
    theta = angle_deg * torch.pi / 180

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # 0°   → [ 1,  0]
    # 90°  → [ 0,  1]
    # 180° → [-1,  0]
    # 270° → [ 0, -1]
    # 360° → [ 1,  0]

    scale = torch.rand(B, device=device) * (1.5 - 0.90) + 0.90 

    # shear_x = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 
    # shear_y = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 

    shear_angle_deg_x = torch.rand(B, device=device) * (17 + 17) - 17
    shear_angle_rad_x = shear_angle_deg_x * torch.pi / 180
    
    shear_angle_deg_y = torch.rand(B, device=device) * (17 + 17) - 17
    shear_angle_rad_y = shear_angle_deg_y * torch.pi / 180

    shear_x = torch.tan(shear_angle_rad_x)
    shear_y = torch.tan(shear_angle_rad_y)
    

    R[:,0] = - dx_norm
    R[:,1] = dy_norm
    R[:,2] = cos_theta
    R[:,3] = sin_theta
    R[:,4] = 1.0 - scale
    R[:,5] = shear_x
    R[:,6] = shear_y

    # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
    T = torch.zeros(B,2,3,device=device)

    T[:,0,0] = cos_theta * scale
    T[:,0,1] = -sin_theta * scale + shear_x
    T[:,1,0] = sin_theta * scale + shear_y
    T[:,1,1] = cos_theta * scale 

    T[:,0,2] = dx_norm
    T[:,1,2] = dy_norm

    grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear',
                            padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R

def Loss_Txt(epoch, NUM_EPOCHS, time, current_loss, RESULTS_DIR_LOSS): 
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    text = f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Time: {time:.2f} seconds; Loss: {current_loss:.4f}\n"
    with open(f'{RESULTS_DIR_LOSS}/Log_Treino.txt', "a", encoding="utf-8") as f:
        f.write(text)
        

def Loss_Txt_Small_Norb(text, RESULTS_DIR_LOSS): 
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    with open(f'{RESULTS_DIR_LOSS}/Log_Treino.txt', "a", encoding="utf-8") as f:
        f.write(text)



################################ Rescontruções Parciais ################################################

def Input_Target_Output_Images(inp, target, out, file_path):
    inp = inp.detach().cpu()
    #save_image(out, f'{file_path}/Reconstruction000.png', nrow=1, padding=1, normalize=False, pad_value=0.5)
    out = torch.sigmoid(out).detach().cpu() # if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()

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
