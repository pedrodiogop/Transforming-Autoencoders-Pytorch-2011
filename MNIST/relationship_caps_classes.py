#   Transformação T1 (aplicada a original image) Original Image -> T1 -> Input Image 
#   Horizontal and vertical translation (pixels) [−2, 2]; 
#   Rotation (degrees) [−15, 15], 
#   Scale [0.95, 1.2]; 
#   Horizontal and vertical shear (degrees) [−10, 10]
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args, set_seed, BatchShift_torch_Rotation
import torch
from CapLayer import CapLayer
import torch.nn as nn
import os
import torchvision
import matplotlib.pyplot as plt
# import torch.nn.functional as F
import pandas as pd


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
            labels_list.append(labels.cpu())
    P = torch.cat(probs_list, dim=1)
    Y = torch.cat(labels_list, dim=0)
    return P, Y

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


def probs_com_transf(model, loader):
    """p de cada cápsula para imagens transformadas (distribuição de treino)."""
    probs_list, labels_list = [], []
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(DEVICE)
            _, dxy, inp_img = BatchShift_torch_Rotation(
                img, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION],
                [-ROTATION_ANGLE, ROTATION_ANGLE], [SCALE_MIN, SCALE_MAX],
                [-SHEAR, SHEAR], padding_mode_sift, DEVICE, LEN_POSE)
            _, prob = model(inp_img, dxy, sep=True)
            probs_list.append(prob.squeeze(-1).cpu())     # [40, B]
            labels_list.append(labels.cpu())
    # print(len(probs_list))
    all_prob_caps = torch.cat(probs_list, dim=1)                       # [40, N_teste]
    classes = torch.cat(labels_list, dim=0)                      # [N_teste]
    return all_prob_caps, classes 

def guardar_resultados(matrix_prob_media, matrix_prob_std, nome, seed, n_classes=10):
    n_caps = matrix_prob_media.shape[0] # numero de capulas
    # n_por_classe = np.array([(Y == c).sum().item() for c in range(n_classes)]) # numero de imagens por classe
    media_por_classe = matrix_prob_media.mean(axis=0)               # [10] média das 40 cápsulas por classe

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(21, 6))
    im = ax.imshow(matrix_prob_media.T, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xlim(-0.5, n_caps + 0.5)                    # espaço para a coluna extra
    ax.set_xticks(list(range(n_caps)) + [n_caps])
    ax.set_xticklabels([str(k) for k in range(n_caps)] + ["Mean"])
    ax.set_yticks(range(n_classes));  ax.set_yticklabels(range(n_classes))
    ax.set_xlabel("Capsule");  ax.set_ylabel("Class")
    ax.set_title(f"Mean ± std of probabilities; {nome}; Seed={seed}")
    fig.colorbar(im, ax=ax)
    #fig.colorbar(im, ax=ax, label="Mean of probabilities")

    # --- valores em cada célula ---
    for c in range(n_classes):                         # linhas = classes
        for k in range(n_caps):                        # colunas = cápsulas
            ax.text(k, c, f"{matrix_prob_media[k, c]:.2f}\n±{matrix_prob_std[k, c]:.2f}",
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white" if matrix_prob_media[k, c] > 0.5 else "black")
        # última coluna: média das frações da classe
        ax.text(n_caps, c, f"{media_por_classe[c]:.2f}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="black")

    ax.axvline(n_caps - 0.5, color="black", linewidth=1)   # separador
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_RELATIONSHIP_CAPS_CLASSES}/Mean_Prob_{nome}_SEED_{seed}.pdf")
    plt.show()


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
    RESULTS_DIR_RELATIONSHIP_CAPS_CLASSES = f'{RESULTS_DIR}/Relationship_Caps_Classes'

    os.makedirs(RESULTS_DIR_RELATIONSHIP_CAPS_CLASSES, exist_ok=True)

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

    # HeapMap 
    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()

    all_prob_caps, vector_classes = probs_com_transf(capL_test, testeloader)
    all_prob_caps_without, _ = probs_sem_transf(capL_test, testeloader)
    # all_prob_caps -> [num_caps, N_teste]
    # vector_classes -> [N_teste]
    _, matrix_prob_media, matrix_prob_std, _ = prob_media(all_prob_caps, vector_classes)
    _, matrix_prob_media_without, matrix_prob_std_without, _ = prob_media(all_prob_caps_without, vector_classes)
    
    guardar_resultados(matrix_prob_media, matrix_prob_std, "With_transformations", SEED)
    guardar_resultados(matrix_prob_media_without, matrix_prob_std_without, "Without_transformations", SEED)