from Class_Norb_teste import SmallNORBPairDataset
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import RandomSampler, DataLoader
import matplotlib.pyplot as plt


# def Save_In_Out_Target_Images(l_image, r_image, categoria):
def Save_In_Out_Target_Images(img):
        os.makedirs('Experimento_Small_Norb', exist_ok=True) 
        batch = torch.cat([l_image, r_image], dim=2)
            
        im_tensor = torchvision.utils.make_grid(batch, nrow=10, normalize=False, padding=2, pad_value=0.5)
        img = np.transpose(im_tensor.numpy(), (1, 2, 0))
        
        diretorio = 'Experimento_Small_Norb/'
        os.makedirs(diretorio, exist_ok=True)
        caminho = os.path.join(diretorio, f'l_r_small_norb_{categoria[0]}_{categoria[1]}_{categoria[2][2]}_{categoria[3][2]}.png')
        plt.imsave(caminho, img)

def Save_In_Out_Target_iluminacao(img):
        os.makedirs('Experimento_Small_Norb', exist_ok=True) 
        batch = torch.cat(img, dim=0)
            
        im_tensor = torchvision.utils.make_grid(batch, nrow=3, normalize=False, padding=2, pad_value=0.5)
        img = np.transpose(im_tensor.numpy(), (1, 2, 0))
        
        diretorio = 'Experimento_Small_Norb/'
        os.makedirs(diretorio, exist_ok=True)
        caminho = os.path.join(diretorio, f'l_r_small_norb_iluminacao.png')
        plt.imsave(caminho, img)

def Save_In_Out_Target_iluminacaoo(img, path='Experimento_Small_Norb/l_r_small_norb_iluminacao.png'):
    """
    img: lista de [img_i, img_t, light_i, light_t]
         img_i, img_t: tensores (1, H, W) em [0, 1]
         light_i, light_t: valor da 3ª componente de attrs_l (iluminação, ou o pixel do fundo)
    Guarda uma figura com a imagem de entrada (i) à esquerda e, a seguir, as imagens alvo (t)
    ordenadas por iluminação, com o valor e a média de cada uma no título.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    #img.append([l_image, r_image, categoria[2][2], categoria[3][2]])
    #   categoria = [i, t, self.attrs_l[i], self.attrs_l[t]]
    # self.attrs_l = [ele, azim, ilumin]
    # ordenar pelas iluminações do alvo
    img = sorted(img, key=lambda e: float(e[3]))

    entrada = img[0][0]                                   # igual em todos (mesmo i)
    alvos   = [e[1] for e in img]
    lights  = [float(e[3]) for e in img]

    tiles  = [entrada] + alvos
    titles = [f"entrada\nluz={float(img[0][2]):.3f}"] + [
        f"luz={l:.3f}\nmédia={a.mean().item():.3f}" for l, a in zip(lights, alvos)
    ]

    fig, axes = plt.subplots(1, len(tiles), figsize=(2.2 * len(tiles), 2.8))
    for ax, im, t in zip(np.atleast_1d(axes), tiles, titles):
        ax.imshow(im.squeeze(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # diferenças em relação à primeira iluminação (alvo)
    ref = alvos[0]
    for l, a in zip(lights, alvos):
        print(f"luz={l:.3f} | média={a.mean():.4f} | mediana={a.median():.4f} | "
              f"|dif vs 1ª|={(a - ref).abs().mean():.4f}")



PATH_DATASET_NORB = 'tmp/data_small_norb'
IMAGE_SIZE = 96
BATCH_SIZE = 64

trainset = SmallNORBPairDataset(
    PATH_DATASET_NORB,
    split='train',
    image_size=IMAGE_SIZE,
    crop_size=80
    )

img = []
for i in range(972, 1944):
    #   categoria = [i, t, self.attrs_l[i], self.attrs_l[t]]
    # self.attrs_l = [ele, azim, ilumin]
    l_image, r_image, categoria = trainset[i]
    if categoria [2][0] == categoria [3][0] and categoria [2][1] == categoria [3][1]:
        img.append([l_image, r_image, categoria[2][2], categoria[3][2]])
Save_In_Out_Target_iluminacaoo(img)


# train_sampler = RandomSampler(trainset, replacement=True, num_samples=10 * 64)
# trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1, pin_memory=True, persistent_workers=True)

# for i, (x, target, tansf) in enumerate(trainloader):
#     Save_In_Out_Target_Images(x, target, i)

    
#info_categoria_l, info_categoria_r = trainset.teste()
#print(info_categoria_l[0:10])
#print()
#print()
#print(info_categoria_r[0:10])
# print(trainset[4][0].max())
# print(trainset[4][0].min())

# print(trainset[20][0].max())
# print(trainset[20][0].min())

#valores_unicos, indices = np.unique(info_categoria[:, 4], return_index=True)


# print(len(trainset)) # 24300
# print(len(trainset[4])) # 4