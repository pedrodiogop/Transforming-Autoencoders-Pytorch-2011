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


def guardar_resultados(frac, Y, thr, nome, seed, n_classes=10):
    """Heatmap (PDF) e tabela de contagens (Excel) para uma condição."""
    n_caps = frac.shape[0] # numero de capulas
    n_por_classe = np.array([(Y == c).sum().item() for c in range(n_classes)]) # numero de imagens por classe
    media_por_classe = frac.mean(axis=0)               # [10] média das 40 cápsulas por classe

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(frac.T, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xlim(-0.5, n_caps + 0.5)                    # espaço para a coluna extra
    ax.set_xticks(list(range(n_caps)) + [n_caps])
    ax.set_xticklabels([str(k) for k in range(n_caps)] + ["Mean"])
    ax.set_yticks(range(n_classes));  ax.set_yticklabels(range(n_classes))
    ax.set_xlabel("Capsule");  ax.set_ylabel("Class")
    ax.set_title(f"Fraction of images with p < {thr}; {nome}; Seed={seed}")
    fig.colorbar(im, ax=ax, label="Inactive Fraction")

    # --- valores em cada célula ---
    for c in range(n_classes):                         # linhas = classes
        for k in range(n_caps):                        # colunas = cápsulas
            ax.text(k, c, f"{frac[k, c]:.2f}", ha="center", va="center", fontsize=6, fontweight="bold",
                    color="white" if frac[k, c] > 0.5 else "black")
        # última coluna: média das frações da classe
        ax.text(n_caps, c, f"{media_por_classe[c]:.2f}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="black")

    ax.axvline(n_caps - 0.5, color="black", linewidth=1)   # separador
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_DEAD_CAPS}/caps_inativas_{nome}_TH_{thr}_SEED_{seed}.pdf")
    plt.show()

    # --- tabela ---
    #df = pd.DataFrame(contagens.T,                     # [10 classes, 40 cápsulas]
                      #index=[f"classe {c}" for c in range(n_classes)],
                      #columns=[f"{k}" for k in range(n_caps)])
    #df["N imagens"] = n_por_classe                     # total por classe, para referência
    #df.to_excel(f"{RESULTS_DIR_DEAD_CAPS}/capsulas_inativas_{nome}.xlsx")
    #return df


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


def contar_inativas(all_prob_caps, vector_classes, thr, n_classes=10):
    """Contagens e frações de imagens com p < thr, por cápsula e classe."""
    contagens = np.zeros((all_prob_caps.shape[0], n_classes), dtype=int)   # [40, 10]
    for k in range(all_prob_caps.shape[0]):
        for c in range(n_classes):
            contagens[k, c] = (all_prob_caps[k, vector_classes == c] < thr).sum().item()
    n_por_classe = np.array([(vector_classes == c).sum().item() for c in range(n_classes)])
    frac = contagens / n_por_classe
    return contagens, frac

def n_mortas(P, Y, thr, frc):
    """Nº de cápsulas com fração inativa = 1 em todas as classes."""
    _, frac = contar_inativas(P, Y, thr)          # frac: [40, 10]
    mortas = (frac >= frc).all(axis=1)            # [40] True se morta
    # print(f" {mortas} dead capsules")
    return int(mortas.sum())


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
    SEED = 19
    SCALE_MIN = args.scale_min
    SCALE_MAX = args.scale_max
    SHEAR = args.shear


    lr = args.lr
    set_seed(SEED)

    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{SCALE_MIN}_{SCALE_MAX}_{SHEAR}_{lr}_{42}_20'
    RESULTS_DIR_DEAD_CAPS = f'{RESULTS_DIR}/Dead_Caps'

    os.makedirs(RESULTS_DIR_DEAD_CAPS, exist_ok=True)

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
    THR = 0.95
    all_prob_caps, vector_classes = probs_com_transf(capL_test, testeloader)
    all_prob_caps_without, _ = probs_sem_transf(capL_test, testeloader)
    # all_prob_caps -> [num_caps, N_teste]
    # vector_classes -> [N_teste]
    _, frac_com = contar_inativas(all_prob_caps, vector_classes, THR)
    #print(contagens_com.shape, frac_com.shape) # (40, 10) (40, 10)
    guardar_resultados(frac_com, vector_classes, THR, "With Transformation", SEED)
    
    # grafico com diferentes thresholds
    THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    frc = 0.90   # fração mínima de inatividade para considerar cápsula morta
    mortas_com = [n_mortas(all_prob_caps, vector_classes, t, frc) for t in THRESHOLDS]
    mortas_sem = [n_mortas(all_prob_caps_without, vector_classes, t, frc) for t in THRESHOLDS]
    # --- gráfico ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(THRESHOLDS, mortas_sem, marker="o", label="without transformations")
    ax.plot(THRESHOLDS, mortas_com, marker="s", label="with transformations")
    ax.set_xticks(THRESHOLDS)
    ax.set_ylim(0, all_prob_caps.shape[0])                    # eixo y de 0 a 40 cápsulas
    ax.set_xlabel("Threshold on P")
    ax.set_ylabel("No. of dead capsules")
    ax.set_title("Threshold sensitivity")
    ax.grid(alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR_DEAD_CAPS}/grafico_sensibilidade_threshold_{SEED}_frc{frc}.pdf")
    plt.show()  







    # P_sem, _ = probs_sem_transf(capL_test, testeloader)
    #contagens_sem, frac_sem = contar_inativas(P_sem, vector_classes, THR)
    #guardar_resultados(contagens_sem, frac_sem, vector_classes, THR, "without transformation", SEED)


    # with torch.no_grad():
    #     for i, (img, labels) in enumerate(testeloader):
    #         img = img.to(DEVICE)
    #         target, dxy, inp_img = BatchShift_torch_Rotation(img, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], [SCALE_MIN, SCALE_MAX], [-SHEAR, SHEAR], padding_mode_sift, DEVICE, LEN_POSE)
    #         _, prob = capL_test(inp_img, dxy, sep = True)
    #         _, prob_without_transf = capL_test(img, without_transf, sep = True)

    #         probs_list.append(prob.squeeze(-1).cpu())   # [40, 256, 1] -> [40, 256]
    #         probs_list_without_transf.append(prob_without_transf.squeeze(-1).cpu())   # [40, 256, 1] -> [40, 256]
    #         labels_list.append(labels.cpu())

    #         # print([tuple(p.shape) for p in probs_list])
    #     P = torch.cat(probs_list, dim=1)   # [num_caps, 256]
    #     P_without_transf = torch.cat(probs_list_without_transf, dim=1)   # [num_caps, 256]
    #     Y = torch.cat(labels_list, dim=0)  # [N_teste]
    #     # print(P.shape, Y.shape) torch.Size([40, 10000]) torch.Size([10000])

    # contagens = np.zeros((P.shape[0], 10), dtype=int)   # [40, 10]
    # for k in range(P.shape[0]):
    #     for c in range(10):
    #         contagens[k, c] = (P[k, Y == c] < THR).sum().item()
    # n_por_classe = np.array([(Y == c).sum().item() for c in range(10)])
    # frac = contagens / n_por_classe  

    # contagens = np.zeros((P_without_transf.shape[0], 10), dtype=int)   # [40, 10]
    # for k in range(P_without_transf.shape[0]):
    #     for c in range(10):
    #         contagens[k, c] = (P_without_transf[k, Y == c] < THR).sum().item()
    # n_por_classe = np.array([(Y == c).sum().item() for c in range(10)])
    # frac = contagens / n_por_classe  

    # for k in range(P.shape[0]):                  # 40 cápsulas
    #     n_inativa = (P[k] < THR).sum().item()    # nº de imagens com p < THR
    #     print(f"cápsula {k:02d}: inativa em {n_inativa}/{P.shape[1]} imagens")

    # for k in range(P.shape[0]):
    #     contagens = []
    #     for c in range(10):
    #         n = (P[k, Y == c] < THR).sum().item()
    #         contagens.append(n)
    #     print(k, contagens)