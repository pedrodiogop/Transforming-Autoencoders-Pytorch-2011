import os
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Get_Args, Save_In_Out_Target_Images, Plot_Loss, Loss_Txt, set_seed, save_summary_to_file
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from Class_Pointing_Head import Class_Pointing04


if __name__ == '__main__':
    args = Get_Args()

    # If you want to see your GPU or CPU in action, you can use the following code to check if PyTorch recognizes it and to set the device accordingly:
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

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

    # Define the directory to save results
    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{IMG_COLOR}_{IMG_W_SIZE}_{IMG_H_SIZE}_{lr}_{SEED}_01'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_DIR_TRAIN = f'{RESULTS_DIR}/Train'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR_TRAIN}/Loss_Image_TXT'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TRAIN}/In_Out_Target_Images'

    trainset = Class_Pointing04(PATH_DATASET_POINTING, IMG_COLOR, IMG_W_SIZE, IMG_H_SIZE, split='train')
    # print("train_dataset: ", len(trainset)) # train_dataset:  129735

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    # print(f"trainloader: {len(trainloader)} batches of size {BATCH_SIZE}") 507 batches of size 256
    # torch.backends.cudnn.benchmark = True  # se o tamanho das imagens for fixo
    
    # images= next(iter(trainloader))
    # print(f"Batch shape: {images.shape}") # → torch.Size([BATCH_SIZE, 3, H, W])
    # plt.imshow(images[0].permute(1, 2, 0))  # primeira imagem do batch
    # plt.axis('off')
    # plt.show()

    sample = trainloader.dataset[0][0]  # (C, H, W)
    # print(sample.shape)  # (1, 32, 32)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    #print(IN_DIM)  # (1, 32, 32)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL = capL.to(DEVICE)
    

    crit = nn.SmoothL1Loss(beta=0.1) # nn.MSELoss() # experimentar SmoothL1Loss
    optimizer = optim.Adam(capL.parameters(), lr)  

    fake_transformation = torch.zeros((BATCH_SIZE, LEN_POSE)).to(DEVICE) 
    fake_img = torch.zeros((BATCH_SIZE, IMG_C, IMG_H, IMG_W)).to(DEVICE)
    model_stats = summary(capL, input_data=[fake_img, fake_transformation], verbose=0)
    save_summary_to_file(model_stats, RESULTS_DIR)

    loss_history = [] # save the loss for each iteration to plot later
    n_batches = len(trainloader) # To save last Input, Output, Target images of each epoch
    best_loss = float('inf')
    capL.train()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        running_loss = torch.zeros((), device=DEVICE)
        n_samples = 0
        for i, (image_A, target, pose_diff) in enumerate(trainloader):
            optimizer.zero_grad()
            image_A = image_A.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            pose_diff = pose_diff.to(DEVICE, non_blocking=True)
            output = capL(image_A, pose_diff).view(-1, IMG_C, IMG_H, IMG_W)
            # aplicar sigmoid a saida e remover o clamp na visualização da imgem
            loss = crit(output, target)
            loss.backward()
            optimizer.step()
            bs = image_A.size(0)
            running_loss += loss.detach() * bs  # multiply by batch size to get total loss for the batch
            n_samples += bs
            if i == n_batches - 2:
                Save_In_Out_Target_Images(image_A, target, output, epoch, RESULTS_DIR_IN_OUT_TARGET_IMAGES)
            if i % 100 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (i + 1) / elapsed
                eta = (n_batches - i - 1) / batches_per_sec
                print(f"\rEpoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{n_batches} "
                    f"| {batches_per_sec:.2f} batch/s | ETA época: {eta:.1f}s",
                    end="", flush=True)

        train_loss = (running_loss / n_samples).item()
        loss_history.append(train_loss)

        # ---------- checkpoint + early stopping (sobre a loss de teste) ----------
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(capL.state_dict(), f'{RESULTS_DIR}/best_model.pth')

        diff_time = time.time() - start_time
        
        text = f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]; Time: {diff_time:.2f}s; Train loss: {train_loss:.5f}; Best: {best_loss:.5f};\n"
        print(text)
        Loss_Txt(text, RESULTS_DIR_LOSS)
        Plot_Loss(loss_history, RESULTS_DIR_LOSS)
    Plot_Loss(loss_history, RESULTS_DIR_LOSS)