from captum.attr import IntegratedGradients
from pyexpat import model
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Get_Args
from CapLayer import CapLayer
from wrapper_ig_presence_prob import PresenceProbWrapper

if __name__ == '__main__':
    args = Get_Args()

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec # encode the image
    CAP_GEN = args.cap_gen # decode the image
    LEN_POSE = args.len_pose
    DATASET = "Face_Equivarience" 
    lr = args.lr
    # IN_DIM = 28 * 28  # MNIST flattened
    # IN_DIM = 32 * 32 * 3  # Face_Equivarience flattened
    IN_DIM = 24 * 18 * 3  # Face_Equivarience flattened

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{lr}_{LEN_POSE}'

    test_dataset  = datasets.CIFAR10(root="tmp", train=False, download=False, transform=ToTensor())
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    X_batch, y_batch = next(iter(testloader))
    X_batch = X_batch.to(DEVICE)# (B, 1, 28, 28) ou equivalente

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    # --- Fase 1: procurar a amostra com maior prob para a cápsula escolhida, em TODO o dataset ---

    capsule_idx = 40
    best_prob = -1.0
    best_image = None
    best_label = None

    capL_test.eval()
    with torch.no_grad():
        for X_b, y_b in testloader:
            X_b = X_b.to(DEVICE)
            delxy_b = torch.zeros(X_b.size(0), LEN_POSE, device=DEVICE)

            _, _, all_probs = capL_test(X_b, delxy_b, sep=True)
            probs_cap = all_probs[capsule_idx].squeeze(-1)  # (B,)

            batch_max_prob, batch_max_idx = probs_cap.max(dim=0)

            if batch_max_prob.item() > best_prob:
                best_prob = batch_max_prob.item()
                best_image = X_b[batch_max_idx].unsqueeze(0).clone()  # (1, C, H, W)
                best_label = y_b[batch_max_idx].item()

    print(f"Melhor amostra encontrada: prob={best_prob:.4f}, label={best_label}")
    # --- Fase 2: correr o IG apenas sobre essa amostra ---

    X_single = best_image.to(DEVICE)                              # (1, C, H, W)
    delxy_single = torch.zeros(1, LEN_POSE, device=DEVICE)
    baseline_single = torch.zeros_like(X_single)

    wrapper = PresenceProbWrapper(capL_test, capsule_idx).to(DEVICE)
    ig = IntegratedGradients(wrapper)

    attributions, delta = ig.attribute(
        X_single,
        baselines=baseline_single,
        n_steps=50,
        additional_forward_args=(delxy_single,),
        return_convergence_delta=True
    )

    print(f"Convergence delta: {delta.abs().item():.6f}")
    # --- Fase 3: visualização (igual à anterior, mas idx=0 porque só há 1 amostra) ---

    attr_img = attributions[0].detach().cpu().numpy()
    orig_img = X_single[0].detach().cpu().numpy()

    attr_map = attr_img.sum(axis=0)              # agrega canais, se CIFAR-10
    orig_img_disp = orig_img.transpose(1, 2, 0).clip(0, 1)  # se for RGB
    # Se for MNIST (1 canal): orig_img_disp = orig_img.squeeze(0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_img_disp)
    axes[0].set_title(f'Original (label={best_label}, prob={best_prob:.2f})')
    axes[0].axis('off')

    vmax = max(abs(attr_map.min()), abs(attr_map.max()))
    im = axes[1].imshow(attr_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'IG - Presença (Cápsula {capsule_idx})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/IG_presence_capsule_{capsule_idx}_best.png', dpi=150)
    plt.show()