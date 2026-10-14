from torch.utils.data import DataLoader
from aux_functions import Get_Args, Save_In_Out_Target_Images, set_seed
import torch
from CapLayer import CapLayer
import torch.nn as nn
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from Class_Pointing_Head import Class_Pointing04

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
    LEN_POSE = args.len_pose
    SEED = args.seed
    DATASET = args.dataset
    PATH_DATASET_POINTING = args.dataset_path
    IMG_H_SIZE = args.img_h_size
    IMG_W_SIZE = args.img_w_size
    IMG_COLOR  = args.img_color

    lr = args.lr
    set_seed(SEED)

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{IMG_COLOR}_{IMG_W_SIZE}_{IMG_H_SIZE}_{lr}_{SEED}_01'
    RESULTS_DIR_TEST = f'{RESULTS_DIR}/Test'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TEST}/In_Out_Target_Images'

    os.makedirs(RESULTS_DIR_TEST, exist_ok=True)

    test_dataset = Class_Pointing04(PATH_DATASET_POINTING, IMG_COLOR, IMG_W_SIZE, IMG_H_SIZE, split='test')
    testeloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    sample = testeloader.dataset[0][0]   # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)

    crit = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()
    test_ssim = 0.0
    test_psnr = 0.0
    running_loss = torch.zeros((), device=DEVICE)
    n_batches = len(testeloader) # To save last Input, Output, Target images of each epoch
    n_samples = 0
    with torch.no_grad():
        capL_test.eval()
        for i, (image_A, target, pose_diff) in enumerate(testeloader):
            image_A = image_A.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            pose_diff = pose_diff.to(DEVICE, non_blocking=True)

            out = capL_test(image_A, pose_diff).view(-1, IMG_C, IMG_H, IMG_W) 
            loss = crit(out, target)
            out = out.clamp(0, 1) # out = torch.sigmoid(out).detach().cpu()  # 

            bs = image_A.size(0)
            running_loss += loss.detach() * bs  # multiply by batch size to get total loss for the batch
            n_samples += bs

            test_ssim += ssim(out, target).item()
            test_psnr += psnr(out, target).item()
            if i % 30 == 0:
                Save_In_Out_Target_Images(image_A, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES)        
        test_loss = (running_loss / n_samples).item()
    print(f"Loss Média  : {test_loss:.4f}")
    print(f"SSIM Médio  : {test_ssim / n_batches:.4f}")
    print(f"PSNR Médio  : {test_psnr / n_batches:.4f} dB")