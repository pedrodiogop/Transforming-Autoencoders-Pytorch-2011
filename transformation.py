import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid, save_image



def BatchShift_torch(imbatch: torch.Tensor, dxdy, angle_range, padding_mode_sift, device, pose_dim):
    B, _, H, W = imbatch.shape

    R = torch.zeros(B, pose_dim, device=device)

    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    tx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
    ty = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    # Generate numbers between - angle_range e angle_range
    angle_deg = (
        torch.rand(B, device=device)
        * (angle_range[1] - angle_range[0])
        + angle_range[0]
    )
    #print(angle_deg)

    # Convert to radianos, because torch only accept that
    theta = angle_deg * torch.pi / 180

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # 0°   → [ 1,  0]
    # 90°  → [ 0,  1]
    # 180° → [-1,  0]
    # 270° → [ 0, -1]
    # 360° → [ 1,  0]

    # A manipulação da dimensao pose sera uma soma + [-0.15, 0.5]
    scale = torch.rand(B, device=device) * (1.5 - 0.90) + 0.90 

    shear_x = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 
    shear_y = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 

    R[:,0] = dx_norm
    R[:,1] = dy_norm
    R[:,2] = cos_theta
    R[:,3] = sin_theta
    R[:,4] = scale
    R[:,5] = shear_x
    R[:,6] = shear_y

    # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
    T = torch.zeros(B,2,3,device=device)

    T[:,0,0] = cos_theta * scale
    T[:,0,1] = -sin_theta * scale + shear_x
    T[:,1,0] = sin_theta * scale + shear_y
    T[:,1,1] = cos_theta * scale 

    T[:,0,2] = dx_norm
    T[:,1,2] = dy_norm

    grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear',
                            padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R, angle_deg, scale


RANDOM_TRANSLATION = 7 #7
ROTATION_ANGLE = 180  #180
# SCALE = 1.5 # 0.9
BATCH_SIZE = 1
# SHEAR = 0.3 # -0.3

if __name__ == '__main__':
    trainset = datasets.MNIST(root="tmp", train=True, download=True, transform=ToTensor())
    image, label = trainset[2]

    #print(image.shape)  # torch.Size([1, 28, 28])
    #print(label)        # classe da imagem, por exemplo 5

    image = image.unsqueeze(0)
    #save_image(image, "Transformation/mnist_image.png")

    target, dxy, angle_deg, scale = BatchShift_torch(image, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], 'zeros', 'cpu', 7)
    print(dxy)  
    grid = make_grid(
        torch.cat([image, target], dim=0),
        nrow=2 
        )   
    save_image(grid, "Transformation/mnist_comparison.png")
    print(angle_deg)
