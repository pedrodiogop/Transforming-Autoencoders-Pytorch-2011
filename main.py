import os
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Get_Args, Plot_Poses, Save_In_Out_Target_Images, BatchShift_torch, Plot_Loss
from gradients_aux import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np

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
    DATASET = args.dataset

    lr = args.lr
    best_loss = 100.0

    # Define the directory to save results
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{lr}'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR}/Image_Loss'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR}/In_Out_Target_Images'
    RESULTS_DIR_POSES = f'{RESULTS_DIR}/Poses'
    # RESULTS_DIR_GRADIENTS = f'{RESULTS_DIR}/Gradients_log.txt'
    RESULTS_DIR_GRADIENTS_MEAN_CAPSULES = f'{RESULTS_DIR}/Mean_Gradients_by_Capsule'
    RESULTS_DIR_GRADIENTS_MEAN_LAYERS = f'{RESULTS_DIR}/Mean_Gradients_by_Layer'


    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR_LOSS, exist_ok=True) # save loss for each epoch
    os.makedirs(RESULTS_DIR_IN_OUT_TARGET_IMAGES, exist_ok=True) # save input, output and target images for each epoch
    os.makedirs(RESULTS_DIR_POSES, exist_ok=True) # save poses for each epoch
    os.makedirs(RESULTS_DIR_GRADIENTS_MEAN_CAPSULES, exist_ok=True) # save gradient flow by capsule for each epoch
    os.makedirs(RESULTS_DIR_GRADIENTS_MEAN_LAYERS, exist_ok=True) # save gradient flow by layer for each epoch

    dataset_class = getattr(datasets, DATASET)
    trainset = dataset_class(root="tmp", train=True, download=True, transform=ToTensor())
    # num_workers is the number of subprocesses to use for data loading. If num_workers is set to 0, the data will be loaded in the main process. 
    # If num_workers is greater than 0, that many subprocesses will be used to load the data in parallel, which can speed up data loading, especially for large datasets. 
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    padding_mode_sift = 'border' if 'CIFAR' in DATASET else 'zeros'
        
    # print(f"train: {len(trainloader.dataset)}")
    # train: 60000
    # print(f"trainloader: {trainloader.dataset[0][0].numel()}")
    # trainloader: 784 (28*28 pixels)

    sample = trainloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    # CapLayer(25, 784, 40, 40)
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_REC)
    capL = capL.to(DEVICE)
    
    # BCEWithLogitsLoss compares the reconstructed image with the SHIFTED image (target)
    # on a pixel-by-pixel basis. The objective is for the network to learn how to 
    # apply the dxy displacement to the original image and reconstruct it 
    # correctly in the new position.
    crit = nn.MSELoss() if 'CIFAR' in DATASET else nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(capL.parameters(), lr)  
    
    # print(capL)
    # CapLayer(
    #   (caps): ModuleList(
    #     (0-24): 25 x Capsule(
    #       (cp): Linear(in_features=784, out_features=40, bias=True)
    #       (xy): Linear(in_features=40, out_features=2, bias=True)
    #       (pr): Linear(in_features=40, out_features=1, bias=True)
    #       (gn): Linear(in_features=2, out_features=40, bias=True)
    #       (rc): Linear(in_features=40, out_features=784, bias=True)
    #     )
    #   )
    # )

    # To check the model architecture 
    # dxy_fake = torch.zeros((BATCH_SIZE, 2)).to(DEVICE) 
    # img_fake = torch.zeros((BATCH_SIZE, 1, 28, 28)).to(DEVICE)
    # summary(capL, input_data=[img_fake, dxy_fake])
    poses = []
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
   
    len_batch_size = len(trainloader) - 2 # To save last Input, Output, Target images of each epoch
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for i, (inp, _) in enumerate(trainloader):
            
            optimizer.zero_grad()

            # inp shape: torch.Size([64, 1, 28, 28])
            inp = inp.to(DEVICE)
            target, dxy = BatchShift_torch(inp, [-4, 4], padding_mode_sift, DEVICE)
            # dxy -> batch of transformations [64, 2]
            # target -> batch of shifted images [64, 1, 28, 28]

            if i != len_batch_size:
                out = capL(inp, dxy) 
            else: 
                out, poses, pose_prob = capL(inp, dxy, sep = True)

            # out -> batch of reconstructed images [64, 784] without sigmoid 
            # R -> batch of poses (x,y) [64, 2]
            out = out.view(-1, IMG_C, IMG_H, IMG_W)
            # out -> batch of reconstructed images [64, 1, 28, 28]
            # target -> batch of shifted images [64, 1, 28, 28]
            loss = crit(out, target)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()

            # Save the loss for plotting later
            loss_history.append(current_loss)
            
            # Save the best model based on the lowest loss
            # Gona use it on test.py
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.clone() for k, v in capL.state_dict().items()}

            # Save the input, output and target images for the first and last batch of each epoch
            if (i == 0 or i == len_batch_size): 
                Save_In_Out_Target_Images(inp, target, out, epoch, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)

            # MEAN GRADIENTS FOR EACH CAPSULE
            grad_flow_caps = Save_Mean_Gradients_by_capsule(capL, grad_flow_caps)
            # MEAN GRADIENTS FOR EACH LAYER 
            grad_flow_layers = Save_Mean_Gradients_by_layer(capL, grad_flow_layers)

        fig, axes = plt.subplots(9, 10, figsize=(12,9))
        fig.suptitle(f'Pesos Generativos rc — Época {epoch+1}', fontsize=6)
        for k in range(9):  # primeiras 9 cápsulas
            weights = capL.caps[k].gen_out.weight.data.cpu()  # (784, 50)
            for p in range(10):  # primeiras 10 dimensões gerativas    
                ax = axes[k][p]
                if IMG_C == 1:
                    ax.imshow(weights[:, p].view(IMG_H, IMG_W), cmap='gray')
                else:
                    w = weights[:, p].view(IMG_H, IMG_W, IMG_C)
                    w = (w - w.min()) / (w.max() - w.min())  # normaliza para [0,1]
                    ax.imshow(w.numpy())
                ax.axis('off')
                if k == 0:
                    ax.set_title(f'Gen {p}', fontsize=4)
        plt.tight_layout()
        os.makedirs('Imagens_Pesos_Generativos', exist_ok=True)
        plt.savefig(f'Imagens_Pesos_Generativos/pesos_rc_ep_{epoch+1}.png')
        plt.close(fig)

    
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]; Time: {(time.time() - start_time):.2f} seconds; Loss: {current_loss:.4f}") 
        
        # Save_Gradients(capL, epoch, RESULTS_DIR_GRADIENTS) # For Future Works 
        Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS, window = 50)

        Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

        # Plot_Poses(poses, RESULTS_DIR_POSES, epoch)
    torch.save(best_state, f'{RESULTS_DIR}/best_model.pth') 