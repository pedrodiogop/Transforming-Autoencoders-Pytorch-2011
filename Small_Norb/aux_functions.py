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

def Get_Args_SmallNorb():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='cuda',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=256,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=400,     help='Number of epochs to train')
    parser.add_argument('--iter_per_epoch',     type=int,   default=10000,     help='Number of iterations per epoch')
    parser.add_argument('--eval_batches',     type=int,   default=1000,     help='Number of batches to evaluate on')
    parser.add_argument('--num_caps',   type=int,   default=100,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=200,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=200,   help='Capsule generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='SmallNorb')
    parser.add_argument('--len_pose',    type=int,   default=3, help='Capsule pose vector length.')
    parser.add_argument('--seed',    type=int,   default=42, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_path', type=str, default='tmp/data_small_norb', help='Path para os ficheiros .mat do smallNORB')
    parser.add_argument('--img_size', type=int, default='96', help='If you want to resize the image')
    parser.add_argument('--p_loss_fuct_fg', type=float, default='0.85', help='Weight for object loss')
    parser.add_argument('--thr', type=float, default='0.1', help='Threshold for pixel difference to consider as object')
    parser.add_argument('--k_kernel', type=int, default='1', help='Size of the kernel for smoothing the mask')

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

def Save_In_Out_Target_Images(inp, target, out, epoch, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch
    
    inp = inp.detach().cpu()
    out = out.clamp(0, 1).detach().cpu() # out = torch.sigmoid(out).detach().cpu()  # 
    target = target.detach().cpu()
    batch = torch.cat([inp, target, out], dim=3)
    
         
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=False, padding=2, pad_value=0.5)
    # To have the real values we need to set normalize=False. 
    # This way the reconstrution image is not manipulated from the original
    # im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5) 
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    # img = np.clip(img, 0, 1) 
    
    caminho = os.path.join(RESULTS_DIR_IN_OUT_TARGET_IMAGES, f'Epoch_{epoch:03d}.png')
    plt.imsave(caminho, img)      

def Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS):
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, label='Loss', linewidth=0.8)
            
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Loss')
    ax.set_title(f'Função de Custo — Época {epoch+1}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{RESULTS_DIR_LOSS}/Loss_Ep_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def Loss_Txt_Small_Norb(text, RESULTS_DIR_LOSS): 
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    with open(f'{RESULTS_DIR_LOSS}/Log_Treino.txt', "a", encoding="utf-8") as f:
        f.write(text)


# def fg_mask(img, thr):
#     #bg = img[:, :, :1, :1] 
#     bg = img.flatten(1).median(dim=1).values.view(-1, 1, 1, 1) # (B, 1, 1, 1) -> median along H*W
#     return ((img - bg).abs() > thr).float()

# def make_mask(x, target, thr, k):
#     with torch.no_grad():
#         #m = torch.maximum(fg_mask(x, thr), fg_mask(target, thr))
#         # stride 1 e padding k//2 para manter o tamanho da máscara igual ao da imagem
#         m = fg_mask(target, thr)
#         m = F.max_pool2d(m, k, stride=1, padding=k // 2) # o k controla a dilatação da máscara, para cobrir melhor o objeto. Tem de ser impar para a a mascara ter o mesmo tamanho da imagem.
#     return m

# def mse_fg_bg(out, target, m, eps=1e-8):
#     se = (out - target) ** 2
#     mse_fg = (se * m).sum() / (m.sum() + eps)
#     mse_bg = (se * (1 - m)).sum() / ((1 - m).sum() + eps)
#     return mse_fg, mse_bg

# def loss_fn(out, target, x, w_fg, thr, k):
#     m = make_mask(x, target, thr, k) # (n, 1, H, W), valores 0/1
#     mse_fg, mse_bg = mse_fg_bg(out, target, m)
#     loss = w_fg * mse_fg + (1 - w_fg) * mse_bg
#     return loss, mse_fg, mse_bg