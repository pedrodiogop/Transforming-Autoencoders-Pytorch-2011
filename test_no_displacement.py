import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import BatchShift_torch, Get_Args, Save_In_Out_Target_Images
import torch
from CapLayer import CapLayer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as T
import torchvision

if __name__ == '__main__':
    args = Get_Args()

    # The architecture and hyperparameters must be the same as those used in training; 
    # the folder name where the model is stored contains all the necessary data.
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec
    CAP_GEN = args.cap_gen
    DATASET = args.dataset
    LEN_POSE = args.len_pose  
    LR = args.lr # Only for directory recognition

    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LR}_{LEN_POSE}'
    RESULTS_DIR_TEST = f'{RESULTS_DIR}/Test'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TEST}/In_Out_Target_Images'


    #os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR_TEST, exist_ok=True)
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch

    dataset_class = getattr(datasets, DATASET)
    test_set = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
    testeloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if 'CIFAR' in DATASET:
        padding_mode_sift = 'reflection' if DEVICE == 'mps' else 'border'
    else:
        padding_mode_sift = 'zeros'

    sample = testeloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)
    crit = nn.MSELoss() if 'CIFAR' in DATASET else nn.BCEWithLogitsLoss() 

    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    
    dxy = torch.zeros(size=(BATCH_SIZE, LEN_POSE), device=DEVICE, dtype=torch.float32) 

    capL_test.eval()
    test_loss = 0.0
    num_batch_size = len(testeloader) - 1
    # with torch.no_grad():
    #     for i, (img, _) in enumerate(testeloader):
    #         img = img.to(DEVICE)
    #         if num_batch_size == i: # the last batch_size can be smaller than the others
    #             dxy = torch.zeros(size=(img.shape[0], LEN_POSE), device=DEVICE, dtype=torch.float32) 
    #         out = capL_test(img, dxy)
    #         out = out.view(-1, IMG_C, IMG_H, IMG_W)
    #         loss = crit(out, img)
    #         Save_In_Out_Target_Images(img, False, out, 1, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)
    #         print(f'Img|Out Save in {RESULTS_DIR_IN_OUT_TARGET_IMAGES}, interaction = {i}/{num_batch_size}')
    #         test_loss += loss.item()

    # print(f"Loss Média no Teste: {test_loss/num_batch_size:.4f}")

    # ── Teste com imagem externa ──────────────────────────────
    IMAGE_PATH = "dogg.jpeg"

    transform_external = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),           # ✅ sem Normalize — igual ao ToTensor() do testeloader
    ])

    img_ext = Image.open(IMAGE_PATH).convert("RGB")
    img_ext_tensor = transform_external(img_ext).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        dxy = torch.zeros(size=(1, LEN_POSE), device=DEVICE, dtype=torch.float32)
        out_ext = capL_test(img_ext_tensor, dxy)
        out_ext = out_ext.view(-1, IMG_C, IMG_H, IMG_W)

    # ✅ usa a mesma função que o loop de teste
    Save_In_Out_Target_Images(img_ext_tensor, False, out_ext, 1, 9999, "aqui", DATASET)
    print(f"Imagem externa guardada em {RESULTS_DIR_IN_OUT_TARGET_IMAGES}")