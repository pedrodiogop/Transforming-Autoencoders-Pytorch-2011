import argparse
import random
# from dateutil import parser
from matplotlib import lines
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
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

# def Get_Args():
#     parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
#     parser.add_argument('--device',     type=str,   default='mps',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
#     parser.add_argument('--batch_size', type=int,   default=64,    help='Batch size for training')
#     parser.add_argument('--epochs',     type=int,   default=40,     help='Number of epochs to train')
#     parser.add_argument('--num_caps',   type=int,   default=25,    help='Number of capsules')
#     parser.add_argument('--cap_rec',    type=int,   default=40,   help='Capsule reconstruction dimension')
#     parser.add_argument('--cap_gen',    type=int,   default=40,   help='Capsule generation dimension')
#     parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
#     parser.add_argument('--dataset',    type=str,   default='MNIST', help='Dataset for training or test, only accepts "MNIST", "FashionMNIST", "CIFAR10".')
#     parser.add_argument('--len_pose',    type=int,   default=2, help='Capsule pose vector length. Use 2 for strict spatial equivariance analysis, or > 2 to prioritize image reconstruction capacity.')
#     parser.add_argument('--size_displacement',    type=int,   default=4, help='To control the size of the displacement, if want to train just for reconstruction set this to 0')
#     parser.add_argument('--custom_dataset', action='store_true', help='Specifically for test.py script. To use this feature, you must create a folder named "Mine_Dataset" inside the folder "Test" and place your custom dataset inside it.')

    
#     return parser.parse_args()

def Get_Args():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='cpu',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=64,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=40,     help='Number of epochs to train')
    parser.add_argument('--num_caps',   type=int,   default=25,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=40,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=40,   help='Capsule generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='MNIST', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'SmallNORB'])
    parser.add_argument('--len_pose',    type=int,   default=6, help='Capsule pose vector length. Use 2 for strict spatial equivariance analysis, or > 2 to prioritize image reconstruction capacity.')
    parser.add_argument('--random_translation',    type=int,   default=4, help='To control the size of the displacement, if want to train just for reconstruction set this to 0')
    parser.add_argument('--rotation_angle',    type=int,   default=30, help='To control range of rotation angles.')
    parser.add_argument('--seed',    type=int,   default=42, help='Random seed for reproducibility.')
    parser.add_argument('--custom_dataset', action='store_true', help='Specifically for test.py script. To use this feature, you must create a folder named "Mine_Dataset" inside the folder "Test" and place your custom dataset inside it.')
    parser.add_argument('--norb_path', type=str, default='./temp/data_small_norb', help='Path para os ficheiros .mat do smallNORB')
    
    return parser.parse_args()

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
    B, C, H, W = imbatch.shape

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

def Loss_Txt(epoch, NUM_EPOCHS, time, current_loss, RESULTS_DIR_LOSS): 
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    text = f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Time: {time:.2f} seconds; Loss: {current_loss:.4f}\n"
    with open(f'{RESULTS_DIR_LOSS}/Log_Treino.txt', "a", encoding="utf-8") as f:
        f.write(text)