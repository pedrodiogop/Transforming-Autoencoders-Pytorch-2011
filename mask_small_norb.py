import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from Class_Small_Norb import SmallNORBPairDataset
from torch.utils.data import DataLoader, RandomSampler



def fg_mask(img, thr=0.1):
    #bg = img[:, :, :1, :1] 
    bg = img.flatten(1).median(dim=1).values.view(-1, 1, 1, 1) # (B, 1, 1, 1) -> median along H*W
    return ((img - bg).abs() > thr).float()

def make_mask(x, target, thr=0.1, k=1):
    with torch.no_grad():
        # m = torch.maximum(fg_mask(x, thr), fg_mask(target, thr))
        # stride 1 e padding k//2 para manter o tamanho da máscara igual ao da imagem
        m = fg_mask(target, thr)
        m = F.max_pool2d(m, k, stride=1, padding=k // 2) # o k controla a dilatação da máscara, para cobrir melhor o objeto. Tem de ser impar para a a mascara ter o mesmo tamanho da imagem.
    return m

def mse_fg_bg(out, target, m, eps=1e-8):
    se = (out - target) ** 2
    mse_fg = (se * m).sum() / (m.sum() + eps)
    mse_bg = (se * (1 - m)).sum() / ((1 - m).sum() + eps)
    return mse_fg, mse_bg

def loss_fn(out, target, x, w_fg=0.5, thr=0.1, k=5):
    m = make_mask(x, target, thr, k) # (n, 1, H, W), valores 0/1
    mse_fg, mse_bg = mse_fg_bg(out, target, m)
    loss = w_fg * mse_fg + (1 - w_fg) * mse_bg
    return loss, mse_fg, mse_bg

@torch.no_grad()
# Save_Mask_Grid(x, target, 0.05, path=f'Experimento_Small_Norb/mascara_thr{thr}.png')
def Save_Mask_Grid(x, target, thr, path='Experimento_Small_Norb/mascara_verificacao.png',
                   n=64 , k=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x = x[:n].detach().cpu()                       # (n, 1, H, W)
    target = target[:n].detach().cpu()             # (n, 1, H, W)
    m = make_mask(x, target, thr, k)               # (n, 1, H, W), valores 0/1

    # sobreposição: target em cinzento, com o objeto tingido de vermelho
    gray = target.repeat(1, 3, 1, 1)               # (n, 3, H, W)
    red = torch.zeros_like(gray)
    red[:, 0] = 1.0
    overlay = gray * (1 - 0.5 * m) + red * (0.5 * m)

    rows = [x.repeat(1, 3, 1, 1),                  # linha 1: input
            target.repeat(1, 3, 1, 1),             # linha 2: target
            m.repeat(1, 3, 1, 1),                  # linha 3: máscara
            overlay]                               # linha 4: máscara sobre o target
    batch = torch.cat(rows, dim=0)                 # (4n, 3, H, W)

    grid = torchvision.utils.make_grid(batch, nrow=n, normalize=False,
                                       padding=2, pad_value=0.5)
    plt.imsave(path, grid.permute(1, 2, 0).numpy())

    frac = m.flatten(1).mean(dim=1)                # fração de píxeis de objeto por imagem
    # print("frac=", frac)
    print(f"thr={thr} k={k} | fração de objeto: "
          f"min={frac.min():.3f} média={frac.mean():.3f} máx={frac.max():.3f}")


trainset = SmallNORBPairDataset(
        'tmp/data_small_norb',
        split='train',
        image_size=32,
        crop_size=80
)
    
# testset  = SmallNORBPairDataset(
#         PATH_DATASET_NORB, 
#         split='test',  
#         image_size=IMAGE_SIZE,
#         crop_size=CROP
# )

# print(len(trainset)) # train_dataset: 47239200
    
train_sampler = RandomSampler(trainset, replacement=True, num_samples=2 * 64)
trainloader = DataLoader(trainset, batch_size=64, sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)


x, target, _ = next(iter(trainloader))

# for thr in (0.05, 0.1, 0.2):
#     Save_Mask_Grid(x, target, path=f'Experimento_Small_Norb/mascara_thr{thr}.png', thr=thr)

thr = 0.1

Save_Mask_Grid(x, target, thr, path=f'Experimento_Small_Norb/mascara_thr{thr}.png')