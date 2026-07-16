import os
from pyexpat import model
import Class_HeadPose
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Get_Args, Save_In_Out_Target_Images, BatchShift_torch, Plot_Loss, PlotGenrative, Loss_Txt
from aux_gradients import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from Class_HeadPose import HeadPoseDataset


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
    DATASET = "Face_Equivarience" 

    lr = args.lr
    best_loss = 100.0

    # Define the directory to save results
    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{lr}_{LEN_POSE}'
    RESULTS_DIR_TRAIN = f'{RESULTS_DIR}/Train'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR_TRAIN}/Loss_Image_TXT'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TRAIN}/In_Out_Target_Images'
    # RESULTS_DIR_POSES = f'{RESULTS_DIR_TRAIN}/Poses'
    # RESULTS_DIR_GRADIENTS = f'{RESULTS_DIR_TRAIN}/Gradients_log.txt'
    RESULTS_DIR_GRADIENTS_MEAN_CAPSULES = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Capsule'
    RESULTS_DIR_GRADIENTS_MEAN_LAYERS = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Layer'
    RESULTS_DIR_GENERATIVE = f'{RESULTS_DIR_TRAIN}/Generative_Plot'
    print(f"Using device: {DEVICE}")


    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch
    # os.makedirs(RESULTS_DIR_POSES, exist_ok=True) # save poses for each epoch
    os.makedirs(RESULTS_DIR_GRADIENTS_MEAN_CAPSULES, exist_ok=True) # save gradient flow by capsule for each epoch
    os.makedirs(RESULTS_DIR_GRADIENTS_MEAN_LAYERS, exist_ok=True) # save gradient flow by layer for each epoch
    os.makedirs(RESULTS_DIR_GENERATIVE, exist_ok=True)

    # 96,72
    # 48 36
    # 24 18 
    trainset = HeadPoseDataset("tmp/HeadPoseImageDatabase", None, 3, 24, 18)
    print("train_dataset: ", len(trainset)) # train_dataset:  259470
    
    # test_dataset  = SmallNORBPairDataset(PATH_DATASET_NORB, split='test',  image_size=32)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    torch.backends.cudnn.benchmark = True  # se o tamanho das imagens for fixo
    
    # # Opção 2 — Via DataLoader (primeiro batch)
    # images= next(iter(trainloader))
    # print(f"Batch shape: {images.shape}") # → torch.Size([BATCH_SIZE, 3, H, W])

    # plt.imshow(images[0].permute(1, 2, 0))  # primeira imagem do batch
    # plt.axis('off')
    # plt.show()

    sample = trainloader.dataset[0][0]  # (C, H, W)
    # print(sample.shape)  # (1, 32, 32)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape
    
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL = capL.to(DEVICE)
    

    crit = nn.MSELoss()
    optimizer = optim.Adam(capL.parameters(), lr)  

    # To check the model architecture 
    # dxy_fake = torch.zeros((BATCH_SIZE, 2)).to(DEVICE) 
    # img_fake = torch.zeros((BATCH_SIZE, 1, 28, 28)).to(DEVICE)
    # summary(capL, input_data=[img_fake, dxy_fake])
    # poses = []
    loss_history = [] # save the loss for each iteration to plot later

    # Initialize dictionaries to store gradient flow data for plotting
    grad_flow_caps = {} 
    grad_flow_layers = {
    'inp_rec': [],
    'rec_xy': [],
    'rec_prob': [],
    'xy_gen': [],
    'gen_out': []
}

    # dxy = torch.zeros(size=(BATCH_SIZE, LEN_POSE), device=DEVICE, dtype=torch.float32) 
    len_batch_size = len(trainloader) - 2 # To save last Input, Output, Target images of each epoch
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        n_batches = len(trainloader)
        for i, (image_A, image_B, pose_diff) in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)
            image_A = image_A.to(DEVICE, non_blocking=True)
            image_B = image_B.to(DEVICE, non_blocking=True)
            pose_diff = pose_diff.to(DEVICE, non_blocking=True)
            output = capL(image_A, pose_diff) 
            output = output.view(-1, IMG_C, IMG_H, IMG_W) 
            loss = crit(output, image_B)
            if i == len_batch_size:
                Save_In_Out_Target_Images(image_A, image_B, output, epoch, i,
                                        RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (i + 1) / elapsed
                eta = (n_batches - i - 1) / batches_per_sec
                print(f"\rEpoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{n_batches} "
                    f"| {batches_per_sec:.2f} batch/s | ETA época: {eta:.1f}s",
                    end="", flush=True)

            current_loss = loss.item()
            loss_history.append(current_loss)
            # if current_loss < best_loss:
            #     best_loss = current_loss
            #     best_state = {k: v.clone() for k, v in capL.state_dict().items()}
            #     torch.save(best_state, f'{RESULTS_DIR}/best_model.pth')

            # grad_flow_caps   = Save_Mean_Gradients_by_capsule(capL, grad_flow_caps)
            # grad_flow_layers = Save_Mean_Gradients_by_layer(capL, grad_flow_layers)
        print()
        # PlotGenrative(epoch, capL, IMG_C, IMG_H, IMG_W, RESULTS_DIR_GENERATIVE, num_capsule=NUM_CAPS, num_generative=CAP_GEN)
        torch.save(capL.state_dict(), f'{RESULTS_DIR}/best_model.pth')
        diff_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Time: {(diff_time):.2f} seconds; Loss: {current_loss:.4f}") 
        Loss_Txt(epoch, NUM_EPOCHS, diff_time, current_loss, RESULTS_DIR_LOSS)
        
        # Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS)

        # Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        # Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

        # Plot_Poses(poses, RESULTS_DIR_POSES, epoch)