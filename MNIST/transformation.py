import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid, save_image

def _sample_T_R(B, H, W, dxdy, angle_range, scale_range, shear_range, device, pose_dim):
    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    tx = torch.randint(low=dxdy[0], high=dxdy[1]+1, size=(B,), device=device).float()
    ty = torch.randint(low=dxdy[0], high=dxdy[1]+1, size=(B,), device=device).float()

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    angle_deg = (
        torch.rand(B, device=device)
        * (angle_range[1] - angle_range[0])
        + angle_range[0]
    )
    angle_normalized = angle_deg / 180.0
    theta = angle_deg * torch.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    scale = torch.rand(B, device=device) * (scale_range[0] - scale_range[1]) + scale_range[1]

    shear_angle_deg_x = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_angle_deg_y = torch.rand(B, device=device) * (shear_range[1] - shear_range[0]) + shear_range[0]
    shear_x = torch.tan(shear_angle_deg_x * torch.pi / 180)
    shear_y = torch.tan(shear_angle_deg_y * torch.pi / 180)

    # ── 3. Vetor de pose ──────────────────────────────────────────────────────
    R = torch.zeros(B, pose_dim, device=device)
    R[:, 0] = - dx_norm
    R[:, 1] = dy_norm
    R[:, 2] = angle_normalized
    R[:, 3] = 1.0 - scale
    R[:, 4] = shear_x
    R[:, 5] = shear_y

    # ── 4. Matriz T ───────────────────────────────────────────────────────────
    T = torch.zeros(B, 2, 3, device=device)
    T[:, 0, 0] = cos_theta * scale
    T[:, 0, 1] = -sin_theta * scale + shear_x
    T[:, 1, 0] = sin_theta * scale + shear_y
    T[:, 1, 1] = cos_theta * scale
    T[:, 0, 2] = dx_norm
    T[:, 1, 2] = dy_norm

    return T, R


def _warp(img, T, padding_mode):
    grid = F.affine_grid(T, img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=False)


def BatchShift_torch_Rotation(imbatch: torch.Tensor,
                              dxdy = [-5, 5], angle_range=[-45, 45], scale_range=[0.85, 1.5], shear_range=[-25, 25],           # imbatch    -> inp_image
                              padding_mode_sift= 'zeros', device='cpu', pose_dim = 6,
                              dxdy_i = [-2, 2], angle_range_i = [-15, 15], scale_range_i = [0.95, 1.2], shear_range_i = [-10, 10],   # inp_image  -> target
                            ):
    B, _, H, W = imbatch.shape

    # imbatch -> inp_image
    T_in, _ = _sample_T_R(B, H, W, dxdy_i, angle_range_i, scale_range_i, shear_range_i,
                       device, pose_dim)
    inp_image = _warp(imbatch, T_in, padding_mode_sift)

    # inp_image -> target 
    T, R = _sample_T_R(B, H, W, dxdy, angle_range, scale_range, shear_range,
                             device, pose_dim)
    target = _warp(inp_image, T, padding_mode_sift)

    return target, R, inp_image 


#RANDOM_TRANSLATION = 0.0 #7
#ROTATION_ANGLE = 30.0  #180
# SCALE = 1.5 # 0.9
# SHEAR = 0.3 # -0.3

if __name__ == '__main__':
    trainset = datasets.MNIST(root="tmp", train=True, download=True, transform=ToTensor())
    image, _ = trainset[0]


    image = image.unsqueeze(0)
    #target, dxy = BatchShift_torch_Rotation(image, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], 'zeros', 'cpu', 7)
    target, dxy, inp_image = BatchShift_torch_Rotation(image)

    grid = make_grid(torch.cat([image, inp_image, target], dim=3), nrow=0)
    save_image(grid, "Transformation/mnist_comparison.png")

    # grid = make_grid(
    #     torch.cat([image01, image03, image05, image07, image02, image, image10, image08, image09, image04], dim=2),
    #     nrow=1 
    #     )   
    
    # save_image(grid, "Transformation/mnist_comparison.png")




# def BatchShift_torch_Rotation(imbatch: torch.Tensor, dxdy, angle_range, padding_mode_sift, device, pose_dim):
#     B, _, H, W = imbatch.shape

#     R = torch.zeros(B, pose_dim, device=device)

#     # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
#     #tx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
#     #ty = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
#     tx = 0.0
#     ty = 0.0    
#     # ── 2. Normalizar translação ──────────────────────────────────────────────
#     dx_norm = tx / (W / 2.0)
#     dy_norm = ty / (H / 2.0)

#     # Generate numbers between - angle_range e angle_range
#     angle_deg = (
#         torch.rand(B, device=device)
#         * (angle_range[1] - angle_range[0])
#         + angle_range[0]
#     )
#     angle_deg = torch.tensor(0.0, device=device, dtype=torch.float32)

#     angle_normalized = angle_deg / 180.0

#     # Convert to radianos, because torch only accept that
#     theta = angle_deg * torch.pi / 180

#     cos_theta = torch.cos(theta)
#     sin_theta = torch.sin(theta)
#     # 0°   → [ 1,  0]
#     # 90°  → [ 0,  1]
#     # 180° → [-1,  0]
#     # 270° → [ 0, -1]
#     # 360° → [ 1,  0]

#     scale = torch.rand(B, device=device) * (1.5 - 0.90) + 0.90 
#     scale = 1.0

#     # shear_x = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 
#     # shear_y = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 

#     #shear_angle_deg_x = torch.rand(B, device=device) * (17 + 17) - 17
#     shear_angle_deg_x = torch.tensor(40.0, device=device, dtype=torch.float32)
#     shear_angle_rad_x = shear_angle_deg_x * torch.pi / 180
    
#     #shear_angle_deg_y = torch.rand(B, device=device) * (17 + 17) - 17
#     shear_angle_deg_y = torch.tensor(0.0, device=device, dtype=torch.float32)
#     shear_angle_rad_y = shear_angle_deg_y * torch.pi / 180

#     shear_x = torch.tan(shear_angle_rad_x)
#     shear_y = torch.tan(shear_angle_rad_y)
    

#     R[:,0] = - dx_norm
#     R[:,1] = dy_norm
#     R[:,2] = angle_normalized 
#     R[:,3] = 1.0 - scale
#     R[:,4] = shear_x
#     R[:,5] = shear_y

#     # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
#     T = torch.zeros(B,2,3,device=device)

#     T[:,0,0] = cos_theta * scale
#     T[:,0,1] = -sin_theta * scale + shear_x
#     T[:,1,0] = sin_theta * scale + shear_y
#     T[:,1,1] = cos_theta * scale  

#     T[:,0,2] = dx_norm
#     T[:,1,2] = dy_norm

#     grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
#     shifted = F.grid_sample(imbatch, grid, mode='bilinear',
#                             padding_mode=padding_mode_sift, align_corners=False)

#     return shifted, R