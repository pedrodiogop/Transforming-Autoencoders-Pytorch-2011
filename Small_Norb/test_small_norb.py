from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args_SmallNorb
import torch
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torchvision
import numpy as np
from Class_Small_Norb import SmallNORBPairDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

def Save_In_Out_Target_Images(inp, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch
    
    inp = inp.detach().cpu()
    out = out.clamp(0, 1).detach().cpu() # out = torch.sigmoid(out).detach().cpu()  # 
    target = target.detach().cpu()
    batch = torch.cat([inp, target, out], dim=3)
    
         
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=False, padding=2, pad_value=0.5)
    # To have the real values we need to set normalize=False. 
    # This way the reconstrution image is not manipulated from the original
    # im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5) 
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    # img = np.clip(img, 0, 1) 
    
    diretorio = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}/Batch_{i:05d}'
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f'batch_{i:05d}.png')
    plt.imsave(caminho, img)

if __name__ == '__main__':
    args = Get_Args_SmallNorb()

    # The architecture and hyperparameters must be the same as those used in training; 
    # The folder name where the model is stored contains all the necessary data.
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec
    CAP_GEN = args.cap_gen
    DATASET = args.dataset
    LEN_POSE = args.len_pose  
    lr = args.lr
    SEED = args.seed
    DATASET = args.dataset 
    PATH_DATASET_NORB = args.dataset_path
    IMAGE_SIZE = args.img_size
    ITERS_PER_EPOCH = args.iter_per_epoch     # nº de batches por "epoch"
    EVAL_BATCHES = args.eval_batches         # nº de batches no teste a cada epoch
    CROP = 80

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{ITERS_PER_EPOCH}_{EVAL_BATCHES}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{IMAGE_SIZE}_{CROP}_{lr}_{SEED}'
    RESULTS_DIR_TEST = f'{RESULTS_DIR}/Test'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TEST}/In_Out_Target_Images'

    os.makedirs(RESULTS_DIR_TEST, exist_ok=True)

    testset  = SmallNORBPairDataset(
        PATH_DATASET_NORB, 
        split='test',  
        image_size=IMAGE_SIZE,
        crop_size=CROP
    )
        
    testeloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(len(testeloader)) # trainloader: 10000

    sample, _, _= testeloader.dataset[0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)

    crit = nn.MSELoss() # nn.BCEWithLogitsLoss() # 

    ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()
    test_loss = 0.0
    test_ssim = 0.0
    test_psnr = 0.0
    num_batch_size = len(testeloader) - 1
    with torch.no_grad():
         for i, (x, target, transf) in enumerate(testeloader):
            img = x.to(DEVICE)
            target = target.to(DEVICE)
            transf = transf.to(DEVICE, non_blocking=True)

            out = capL_test(img, transf).view(-1, IMG_C, IMG_H, IMG_W)
            loss = crit(out, target)
            ssim_score = ssim(out, target)
            psnr_score = psnr(out, target)

            test_loss += loss.item()
            test_ssim += ssim_score.item()
            test_psnr += psnr_score.item()

            Save_In_Out_Target_Images(img, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)

    print(f"Loss Média  : {test_loss / len(testeloader):.4f}")
    print(f"SSIM Médio  : {test_ssim / len(testeloader):.4f}")
    print(f"PSNR Médio  : {test_psnr / len(testeloader):.4f} dB")