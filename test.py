from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args, BatchShift_torch, set_seed
# from Class_HeadPose import CustomImageDataset
import torch
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torchvision
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import torch.nn.functional as F


def BatchShift_torch_Rotation(imbatch: torch.Tensor, dxdy, angle_range, padding_mode_sift, device, pose_dim):
    B, _, H, W = imbatch.shape

    R = torch.zeros(B, pose_dim, device=device)

    # ── 1. Amostrar parâmetros ────────────────────────────────────────────────
    tx = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()
    ty = torch.randint(low=dxdy[0], high=dxdy[1], size=(B,), device=device).float()

    # ── 2. Normalizar translação ──────────────────────────────────────────────
    dx_norm = tx / (W / 2.0)
    dy_norm = ty / (H / 2.0)

    # # Generate numbers between - angle_range e angle_range
    # angle_deg = (
    #     torch.rand(B, device=device)
    #     * (angle_range[1] - angle_range[0])
    #     + angle_range[0]
    # )

    angle_deg = torch.tensor([50.0], device=device)  

    angle_normalized = angle_deg / 180.0

    # Convert to radianos, because torch only accept that
    theta = angle_deg * torch.pi / 180

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # 0°   → [ 1,  0]
    # 90°  → [ 0,  1]
    # 180° → [-1,  0]
    # 270° → [ 0, -1]
    # 360° → [ 1,  0]

    scale = torch.rand(B, device=device) * (1.5 - 0.90) + 0.90 

    # shear_x = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 
    # shear_y = torch.rand(B, device=device) * (0.30 + 0.30) - 0.30 

    shear_angle_deg_x = torch.rand(B, device=device) * (17 + 17) - 17
    shear_angle_rad_x = shear_angle_deg_x * torch.pi / 180
    
    shear_angle_deg_y = torch.rand(B, device=device) * (17 + 17) - 17
    shear_angle_rad_y = shear_angle_deg_y * torch.pi / 180

    shear_x = torch.tan(shear_angle_rad_x)
    shear_y = torch.tan(shear_angle_rad_y)
    

    R[:,0] = 0 # dx_norm
    R[:,1] = 0 #dy_norm
    R[:,2] = angle_normalized 
    #R[:,3] = sin_theta
    R[:,3] = 0 # scale - 1.0
    R[:,4] = 0 # shear_x
    R[:,5] = 0 # shear_y

    # ── 4. Construir T e aplicar à imagem ────────────────────────────────────
    T = torch.zeros(B,2,3,device=device)

    # T[:,0,0] = cos_theta * scale
    # T[:,0,1] = -sin_theta * scale + shear_x
    # T[:,1,0] = sin_theta * scale + shear_y
    # T[:,1,1] = cos_theta * scale 

    # T[:,0,2] = dx_norm
    # T[:,1,2] = dy_norm

    T[:,0,0] = cos_theta
    T[:,0,1] = -sin_theta
    T[:,1,0] = sin_theta 
    T[:,1,1] = cos_theta  

    T[:,0,2] = 0
    T[:,1,2] = 0

    grid    = F.affine_grid(T, imbatch.size(), align_corners=False)
    shifted = F.grid_sample(imbatch, grid, mode='bilinear',
                            padding_mode=padding_mode_sift, align_corners=False)

    return shifted, R

def Save_In_Out_Target_Images(inp, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    inp = inp.detach().cpu()
    out = torch.sigmoid(out).detach().cpu() if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()

    if target is not False and target is not None:
        target = target.detach().cpu()
        batch = torch.cat([inp, target, out], dim=3)
    else: 
        batch = torch.cat([inp, out], dim=3)
         
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=False, padding=2, pad_value=0.5)
    # To have the real values we need to set normalize=False. 
    # This way the reconstrution image is not manipulated from the original
    # im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5) 
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    # img = np.clip(img, 0, 1) 
    
    diretorio = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}'
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f'batch_{i:05d}.png')
    plt.imsave(caminho, img)

# If you want to use Custom_Dataset, you need to create a folder named "Mine_Dataset" inside the folder "Test" and place your custom dataset inside it.
# And pass in the command line argument --custom_dataset when running the test script.

if __name__ == '__main__':
    args = Get_Args()

    # The architecture and hyperparameters must be the same as those used in training; 
    # The folder name where the model is stored contains all the necessary data.
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
    SEED = args.seed
    CUSTOM_DATASET = args.custom_dataset
    SEED = args.seed

    lr = args.lr
    set_seed(SEED)


    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{lr}_{SEED}'
    RESULTS_DIR_TEST = f'{RESULTS_DIR}/Test'
    # RESULTS_DIR_MINE_DATA_SET = f'{RESULTS_DIR_TEST}/Mine_Dataset'
    # RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITH_DISPLACEMENT = f'{RESULTS_DIR_TEST}/Results_Mine_Test_With_Displacement'
    # RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITHOUT_DISPLACEMENT = f'{RESULTS_DIR_TEST}/Results_Mine_Test_Without_Displacement'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITH_TRANSFORMATION = f'{RESULTS_DIR_TEST}/In_Out_Target_Images_With_Transformation'
    # RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT = f'{RESULTS_DIR_TEST}/In_Out_Target_Images_Without_Displacement'

    os.makedirs(RESULTS_DIR_TEST, exist_ok=True)

    ###### ATENTION ######
    # The hyperparameters RANDOM_DISPLACEMENT needs to be the same as the folder name where the model is stored, because it controls the path.
    # If you want to customize the displacement 
    # Uncomment the follow line and set as you wish
    # RANDOM_DISPLACEMENT = 0 # if you want to test only the reconstruction capacity of the model, set this to 0.

    # if 'CIFAR' in DATASET:
    #     padding_mode_sift = 'reflection' if DEVICE == 'mps' else 'border'
    # else:
    padding_mode_sift = 'zeros'    

    # if not CUSTOM_DATASET: # Standard Dataset
    dataset_class = getattr(datasets, DATASET)
    test_set = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
    # else: # Custom Dataset
    #     if 'CIFAR' in DATASET: # TO DEFINE THE SHAPE OF THE IMAGES IN THE CUSTOM DATASET
    #         IMG_C, IMG_H, IMG_W = 3, 32, 32
    #     else: # MNIST or FashionMNIST TO DEFINE THE SHAPE OF THE IMAGES IN THE CUSTOM DATASET 
    #         IMG_C, IMG_H, IMG_W = 1, 28, 28
    #     test_set = CustomImageDataset(
    #         folder_path= RESULTS_DIR_MINE_DATA_SET,
    #         img_c= IMG_C,
    #         img_h= IMG_H,
    #         img_w= IMG_W,
    #         transform= ToTensor()
    #         )
        
    testeloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    sample = testeloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    # crit = nn.MSELoss() if 'CIFAR' in DATASET else nn.BCEWithLogitsLoss() 
    crit = nn.BCEWithLogitsLoss() 
    ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()
    test_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
    num_batch_size = len(testeloader) - 1
    with torch.no_grad():
        for i, (img, _) in enumerate(testeloader):
            img = img.to(DEVICE)
            
            if RANDOM_TRANSLATION != 0: # with displacement 
                # target, dxy = BatchShift_torch(img, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], padding_mode_sift, DEVICE, LEN_POSE)
                target, dxy = BatchShift_torch_Rotation(img, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], padding_mode_sift, DEVICE, LEN_POSE)
                out = capL_test(img, dxy)
                out = out.view(-1, IMG_C, IMG_H, IMG_W)
                loss = crit(out, target)
                out_for_metrics = torch.sigmoid(out) if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1)
                ssim_score = ssim(out_for_metrics, target)
                psnr_score = psnr(out_for_metrics, target)
                test_ssim += ssim_score.item()
                test_psnr += psnr_score.item()
                
                if not CUSTOM_DATASET: # Standard Dataset
                    Save_In_Out_Target_Images(img, target, out, i, f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITH_TRANSFORMATION}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}', DATASET)
                    print(f'Img|Out Save in {RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITH_TRANSFORMATION}, interaction = {i}/{num_batch_size}')
                else:
                    Save_In_Out_Target_Images(img, target, out, i, f'{RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITH_TRANSFORMATION}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}', DATASET)
                    print(f'Img|Out Save in {RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITH_TRANSFORMATION}, interaction = {i}/{num_batch_size}')

            # else: # Analyze only images reconstruction without displacement.
            #     dxy = torch.zeros(size=(img.shape[0], LEN_POSE), device=DEVICE, dtype=torch.float32) 
            #     out = capL_test(img, dxy)
            #     out = out.view(-1, IMG_C, IMG_H, IMG_W) 
            #     loss = crit(out, img)
            #     ssim_score = ssim(out, img)
            #     psnr_score = psnr(out, img)
            #     test_ssim += ssim_score.item()
            #     test_psnr += psnr_score.item()
            #     if not CUSTOM_DATASET: # Standard Dataset
            #         Save_In_Out_Target_Images(img, False, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT, DATASET)
            #         print(f'Img|Out Save in {RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT}, interaction = {i}/{num_batch_size}')
            #     else:
            #         Save_In_Out_Target_Images(img, False, out, i, RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITHOUT_DISPLACEMENT, DATASET)
            #         print(f'Img|Out Save in {RESULTS_DIR_MINE_TEST_IN_OUT_TARGET_WITHOUT_DISPLACEMENT}, interaction = {i}/{num_batch_size}')
            
            test_loss += loss.item()

        print(f"Loss Média no Teste: {test_loss/len(testeloader):.4f}")
        print(f"SSIM Média no Teste: {test_ssim/len(testeloader):.4f}")
        print(f"PSNR Média no Teste: {test_psnr/len(testeloader):.4f}")