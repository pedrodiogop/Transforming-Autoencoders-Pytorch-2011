from cv2 import warpAffine
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

def Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS, window):
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, label='Loss', linewidth=0.8)
            
    # Média móvel para ver a tendência
    if len(loss_history) >= window: # The real trend, without the noise.
        moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(loss_history)), moving_avg, 
                label=f'Média móvel ({window})', color='red', linewidth=1.5)
            
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Loss')
    ax.set_title(f'Função de Custo — Época {epoch+1}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f'{RESULTS_DIR_LOSS}/loss_ep_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def BatchShift_torch(imbatch: torch.Tensor, dxdy, device='cpu'):
    """
    Substitui o BatchShift original — corre 100% no MPS/CUDA/CPU sem sair da GPU.
    
    imbatch: tensor shape (B, 1, 28, 28) já no device
    dxdy: intervalo de deslocamentos em píxeis
    Retorna: (shifted_batch, R) ambos tensores no mesmo device
    """
    B, C, H, W = imbatch.shape

    # Deslocamentos aleatórios em píxeis → normalizar para [-1, 1]
    R = torch.randint(low=dxdy[0], high=dxdy[1], size=(B, 2), device=device).float()
    
    # grid_sample espera deslocamentos normalizados
    dx_norm = R[:, 0] / W  # deslocamento horizontal normalizado
    dy_norm = R[:, 1] / H  # deslocamento vertical normalizado

    # Matriz de transformação afim: [[1, 0, dx], [0, 1, dy]] para cada imagem
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = 1.0         # escala X
    theta[:, 1, 1] = 1.0         # escala Y
    theta[:, 0, 2] = dx_norm     # translação X
    theta[:, 1, 2] = dy_norm     # translação Y

    # Gera a grelha de sampling e aplica o shift
    grid = F.affine_grid(theta, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear', 
                             padding_mode='zeros', align_corners=False)

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

# (imbatch[i], R[i][0], R[i][1])
def shift(img, dx, dy):
    img = np.transpose(img, (1,2,0))
    r,c,_ = img.shape
#    print(r,c)
    trans = np.array([[1,0,dx],[0,1,dy]]).astype(np.float32)
    wimg = warpAffine(img, trans, (r,c)).reshape(r,c,1)
#    print(wimg.shape)
    rimg = np.transpose(wimg, (2,0,1))
#    print(rimg.shape)
    return rimg

# (inp.numpy().copy(),[-4,4])
# lembrar que imbatch tem shape (16,1,28,28) e esta dentro de um for 
def BatchShift(imbatch, dxdy = [-4,4]):
    dim = imbatch.shape
#    print(dim)
#    r = dim[2]
#    c = dim[3]
    imbatch_shift = imbatch.copy()
    # crias um array de deslocamentos aleatórios para cada imagem do lote, onde cada deslocamento é um par de valores (dx, dy) gerados aleatoriamente dentro do intervalo especificado por dxdy. O número de deslocamentos gerados é igual ao número de imagens no lote (dim[0]).
    R = np.random.randint(low=dxdy[0], high=dxdy[1], size=(dim[0],2))    
    #print(R.shape) (16,2) para cada imagem do lote temos um par de deslocamentos (dx, dy)
    for i in range(dim[0]):
        imbatch_shift[i] = shift(imbatch[i], R[i][0], R[i][1])
        # para ver as imagens origianis e deslocadas save_shifted_images
        #save_shifted_images(imbatch[i], imbatch_shift[i], epoch, batch, img_idx=i) # Salva as imagens deslocadas para verificação
    return imbatch_shift, R

def Show_Batch(inp, target, out, epoch, i, title, save=True):
    C, H, W = inp.shape[1], inp.shape[2], inp.shape[3]  # infere do batch
    inp    = inp.detach().cpu().view(-1, C, H, W)
    target = target.detach().cpu().view(-1, C, H, W)
    out    = torch.sigmoid(out).detach().cpu().view(-1, C, H, W)
    
    batch = torch.cat([inp, target, out], dim=3)
    im = torchvision.utils.make_grid(batch, nrow=8, normalize=True)
    img_exibir = np.transpose(im.numpy(), (1, 2, 0))
    
    if save:
        diretorio = 'images/{0}/epoch_{1:02d}'.format(title, epoch)
        os.makedirs(diretorio, exist_ok=True)
        caminho = os.path.join(diretorio, 'batch_{0:05d}.png'.format(i))
        plt.imsave(caminho, img_exibir)