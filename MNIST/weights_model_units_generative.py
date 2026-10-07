import torch
from CapLayer import CapLayer
from aux_functions import Get_Args
import matplotlib.pyplot as plt


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
                    #if img_c == 1:
                    ax.imshow(weights[:, p].view(img_h, img_w, img_c), cmap='RdBu_r')
                    #else: # Cifar
                    #    w = weights[:, p].view(img_h, img_w, img_c)
                    #    #w = (w - w.min()) / (w.max() - w.min())  # normaliza para [0,1]
                    #    ax.imshow(w.numpy())
                    ax.axis('off')
                    if k == 0:
                        ax.set_title(f'Generative {p}', fontsize=4)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # em vez de tight_layout
            plt.savefig(f'{RESULTS_DIR_GENERATIVE}/weight_gen_out_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

if __name__ == '__main__':
    args = Get_Args()
    DEVICE = "cpu"
    BATCH_SIZE = 128
    NUM_CAPS = 100
    CAP_REC = 50
    CAP_GEN = 50
    DATASET = "Face_Equivarience"
    #DATASET = "MNIST"
    LEN_POSE = 2
    LR = 0.001
    IN_DIM = 1296
    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LR}_{LEN_POSE}'
    RESULTS_DIR_TRAIN = f'{RESULTS_DIR}/Train'
    RESULTS_DIR_GENERATIVE = f'{RESULTS_DIR_TRAIN}'

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    PlotGenrative(23, capL_test, 3, 24, 18, RESULTS_DIR_GENERATIVE, num_capsule=NUM_CAPS, num_generative=CAP_GEN)