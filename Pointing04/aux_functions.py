import argparse
import random
from matplotlib import lines
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

def Get_Args():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='cuda',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=256,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=500,     help='Number of epochs to train')
    parser.add_argument('--eval_batches',     type=int,   default=500,     help='Number of batches to evaluate on')
    parser.add_argument('--num_caps',   type=int,   default=100,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=200,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=200,   help='Capsule generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='Pointing_04', help='Dataset')
    parser.add_argument('--len_pose',    type=int,   default=2, help='Capsule pose vector length.')
    parser.add_argument('--seed',    type=int,   default=42, help='Random seed for reproducibility.')
    parser.add_argument('--dataset_path', type=str, default='tmp/Pointing04', help='Path para as imagens do dataset')
    parser.add_argument('--img_w_size', type=int, default=67, help='ex: 96 or 67 or 48 or 24; If you want to resize the image')
    parser.add_argument('--img_h_size', type=int, default=50, help='ex: 72 or 50 or 36 or 18; If you want to resize the image')
    parser.add_argument('--img_color', type=str, default='RGB', help='RGB pfor color images, L for grayscale')

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

def Save_In_Out_Target_Images(inp, target, out, epoch, results_dir_in_out_target_images):
    os.makedirs(results_dir_in_out_target_images, exist_ok=True) # save input, output and target images for each epoch
    
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
    
    caminho = os.path.join(results_dir_in_out_target_images, f'Epoch_{epoch:03d}.png')
    plt.imsave(caminho, img)      

def Plot_Loss(loss_history, RESULTS_DIR_LOSS):
    epochs = range(1, len(loss_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, loss_history, label='Treino', marker='o', markersize=3, linewidth=1.0)

    ax.set_xlabel('Época')
    ax.set_ylabel('MSE (média por píxel)')
    ax.set_title('Função de custo por época')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(f'{RESULTS_DIR_LOSS}/Loss_Curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def Loss_Txt(text, RESULTS_DIR_LOSS): 
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    with open(f'{RESULTS_DIR_LOSS}/Log_Treino.txt', "a", encoding="utf-8") as f:
        f.write(text)