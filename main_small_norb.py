import os
from pyexpat import model
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Get_Args, Save_In_Out_Target_Images, BatchShift_torch, Plot_Loss, PlotGenrative, Loss_Txt
from gradients_aux import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from SmallNORBPairDataset import SmallNORBPairDataset


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
    PATH_DATASET_NORB = "tmp/data_small_norb"
    DATASET = "SmallNORB" 

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


    trainset = SmallNORBPairDataset(
        PATH_DATASET_NORB,
        split='train',
        image_size=32,
        pair_mode='azimuth', # escolher entre azimuth ou elevation PENSO QUE VAMOS TER DE ALTERAR PARA ESTE PARAMETRO POIS ASSIM APENAS ESTAMOS A ANALISAR O AZIMUTH OU ELAVATION E TAMBEM QUERO A ILUMINAÇÃO
        max_delta=1          # pares com diferença de 1 step = 20° de azimute
    )
    # print("train_dataset: ", len(trainset)) # train_dataset:  22950
    
    # test_dataset  = SmallNORBPairDataset(PATH_DATASET_NORB, split='test',  image_size=32)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    sample = trainloader.dataset[0]  # (C, H, W)
    # print(sample.shape)  # (1, 32, 32)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape
    
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL = capL.to(DEVICE)
    

    crit = nn.BCEWithLogitsLoss() 
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
        # for i, (image_A, image_B, T, params) in enumerate(trainloader):
        for i, image_A in enumerate(trainloader):
            

            optimizer.zero_grad()

            image_A = image_A.to(DEVICE)
            # print(image_A.size())
            # image_B = image_B.to(DEVICE)
            R = torch.zeros(image_A.size(0), LEN_POSE, device=DEVICE, dtype=torch.float32)
            # R[:, :6] = T.view(T.size(0), 6)

            output = capL(image_A, R)             
            output = output.view(-1, IMG_C, IMG_H, IMG_W) # (B, 1, 32, 32)

            loss = crit(output, image_A)
            # Valores entre 0 e 1 — reflete melhor a perceção humana
            # > 0.8 é considerado bom


            if i == len_batch_size:
                Save_In_Out_Target_Images(image_A, False, output, epoch, i,
                                        RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()

            loss_history.append(current_loss)

            #if current_loss < best_loss:
            best_loss = current_loss
            best_state = {k: v.clone() for k, v in capL.state_dict().items()}
            torch.save(best_state, f'{RESULTS_DIR}/best_model.pth')

            # grad_flow_caps   = Save_Mean_Gradients_by_capsule(capL, grad_flow_caps)
            # grad_flow_layers = Save_Mean_Gradients_by_layer(capL, grad_flow_layers)

        # PlotGenrative(epoch, capL, IMG_C, IMG_H, IMG_W, RESULTS_DIR_GENERATIVE, num_capsule= NUM_CAPS, num_generative=CAP_GEN)
        
        diff_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Time: {(diff_time):.2f} seconds; Loss: {current_loss:.4f}") 
        Loss_Txt(epoch, NUM_EPOCHS, diff_time, current_loss, RESULTS_DIR_LOSS)
        
        # Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS)

        # Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        # Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

        # Plot_Poses(poses, RESULTS_DIR_POSES, epoch)