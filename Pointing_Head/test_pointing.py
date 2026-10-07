from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args
import torch
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torchvision
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from Class_HeadPose import HeadPoseDataset

def Save_In_Out_Target_Images(inp, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    inp = inp.detach().cpu()
    out = out.clamp(0, 1).detach().cpu()

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


if __name__ == '__main__':
    args = Get_Args()

    # The architecture and hyperparameters must be the same as those used in training; 
    # The folder name where the model is stored contains all the necessary data.
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec
    CAP_GEN = args.cap_gen
    DATASET = "Face_Equivarience"
    LEN_POSE = args.len_pose  
    LR = args.lr  

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LR}_{LEN_POSE}'
    RESULTS_DIR_TEST = f'{RESULTS_DIR}/Test'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT = f'{RESULTS_DIR_TEST}/In_Out_Target_Images'

    os.makedirs(RESULTS_DIR_TEST, exist_ok=True)
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT, exist_ok=True)  

    test_dataset = HeadPoseDataset("tmp/test", None, 3, 24, 18)
        
    testeloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    torch.backends.cudnn.benchmark = True  # se o tamanho das imagens for fixo
    sample = testeloader.dataset[0][0]   # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    crit = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()
    test_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
    # num_batch_size = len(testeloader) - 1
    with torch.no_grad():
          for i, (image_A, image_B, pose_diff) in enumerate(testeloader):
            image_A = image_A.to(DEVICE, non_blocking=True)
            image_B = image_B.to(DEVICE, non_blocking=True)
            pose_diff = pose_diff.to(DEVICE, non_blocking=True)

            out = capL_test(image_A, pose_diff)

            out = out.view(-1, IMG_C, IMG_H, IMG_W) 
            loss = crit(out, image_B)
            ssim_score = ssim(out, image_B)
            psnr_score = psnr(out, image_B)

            test_loss += loss.item()
            test_ssim += ssim_score.item()
            test_psnr += psnr_score.item()
            
            Save_In_Out_Target_Images(image_A, image_B, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES_WITHOUT_DISPLACEMENT, DATASET)
            print(f'Batch {i}/{len(testeloader)} | Loss = {loss.item():.4f} | SSIM = {ssim_score:.4f} | PSNR = {psnr_score:.4f} dB')

    print(f"Loss Média  : {test_loss / len(testeloader):.4f}")
    print(f"SSIM Médio  : {test_ssim / len(testeloader):.4f}")
    print(f"PSNR Médio  : {test_psnr / len(testeloader):.4f} dB")