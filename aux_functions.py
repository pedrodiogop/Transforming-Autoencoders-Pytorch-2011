import argparse

from cv2 import warpAffine
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

def Get_Args():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='mps',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=64,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=40,     help='Number of epochs to train')
    parser.add_argument('--num_caps',   type=int,   default=25,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=40,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=40,   help='Generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='MNIST', help='Dataset to use for training, only accepts "MNIST", "FashionMNIST" or "CIFAR10".')
    
    return parser.parse_args()

def PlotGenrative(epoch, capL, img_c, img_h, img_w, RESULTS_DIR_GENERATIVE, num_capsule, num_generative):
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

    
def Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS, window):
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, label='Loss', linewidth=0.8)
            
    if len(loss_history) >= window: # The real trend, without the noise.
        moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(loss_history)), moving_avg, 
                label=f'Média móvel ({window})', color='red', linewidth=1.5)
            
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Loss')
    ax.set_title(f'Função de Custo — Época {epoch+1}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{RESULTS_DIR_LOSS}/Loss_Ep_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def Save_In_Out_Target_Images(inp, target, out, epoch, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    inp = inp.detach().cpu()
    target = target.detach().cpu()
    out = torch.sigmoid(out).detach().cpu() if 'CIFAR' not in DATASET else out.clamp(0, 1).detach().cpu()

    batch = torch.cat([inp, target, out], dim=3)
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5)
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    
    diretorio = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}/Epoch_{epoch:03d}'
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f'batch_{i:05d}.png')
    plt.imsave(caminho, img)

def BatchShift_torch(imbatch: torch.Tensor, dxdy, padding_mode_sift, device):

    B, C, H, W = imbatch.shape
    R = torch.randint(low=dxdy[0], high=dxdy[1], size=(B, 2), device=device).float()
    
    dx_norm = R[:, 0] / W 
    dy_norm = R[:, 1] / H 

    theta = torch.zeros(B, 2, 3, device=device)

    theta[:, 0, 0] = 1.0         # escala X
    theta[:, 1, 1] = 1.0         # escala Y
    theta[:, 0, 2] = dx_norm     # translação X
    theta[:, 1, 2] = dy_norm     # translação Y

    grid = F.affine_grid(theta, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear', 
                             padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R

# Função que guarda as imagens originais e deslocadas para estudo! 
# def save_shifted_images(img, rimg, epoch, batch_idx, img_idx):
#     folder_path = f"images/shift/epoch{epoch}_batch_size{batch_idx}"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     img = np.transpose(img, (1,2,0))
#     rimg = np.transpose(rimg, (1,2,0))
        
#     if img.max() <= 1.0:
#         img = (img * 255).astype(np.uint8)

#     if rimg.max() <= 1.0:
#         rimg = (rimg * 255).astype(np.uint8)

#     file_path = os.path.join(folder_path, f"{img_idx}.png")
#     file_path_shifted = os.path.join(folder_path, f"{img_idx}_shifted.png")
#     cv2.imwrite(file_path, img)
#     cv2.imwrite(file_path_shifted, rimg)
