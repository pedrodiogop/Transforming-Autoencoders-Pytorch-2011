import os
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from aux_functions import Get_Args, Save_In_Out_Target_Images, BatchShift_torch, Plot_Loss, PlotGenrative, Loss_Txt, set_seed, save_summary_to_file
from aux_gradients import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time

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
    LEN_POSE = args.len_pose
    RANDOM_TRANSLATION = args.random_translation
    ROTATION_ANGLE = args.rotation_angle
    SEED = args.seed

    print(DEVICE)
    lr = args.lr
    best_loss = 100.0
    set_seed(SEED)

    # Define the directory to save results
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{lr}_{SEED}'
    RESULTS_DIR_TRAIN = f'{RESULTS_DIR}/Train'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR_TRAIN}/Loss_Image_TXT'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TRAIN}/In_Out_Target_Images'
    # RESULTS_DIR_POSES = f'{RESULTS_DIR_TRAIN}/Poses'
    # RESULTS_DIR_GRADIENTS = f'{RESULTS_DIR_TRAIN}/Gradients_log.txt'
    RESULTS_DIR_GRADIENTS_MEAN_CAPSULES = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Capsule'
    RESULTS_DIR_GRADIENTS_MEAN_LAYERS = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Layer'
    RESULTS_DIR_GENERATIVE = f'{RESULTS_DIR_TRAIN}/Generative_Plot'

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # os.makedirs(RESULTS_DIR_POSES, exist_ok=True) # save poses for each epoch

    dataset_class = getattr(datasets, DATASET)
    trainset = dataset_class(root="tmp", train=True, download=True, transform=ToTensor())
    # num_workers is the number of subprocesses to use for data loading. If num_workers is set to 0, the data will be loaded in the main process. 
    # If num_workers is greater than 0, that many subprocesses will be used to load the data in parallel, which can speed up data loading, especially for large datasets. 
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    torch.backends.cudnn.benchmark = True  # se o tamanho das imagens for fixo
    

    # if 'CIFAR' in DATASET:
    #     padding_mode_sift = 'reflection' if DEVICE == 'mps' else 'border'
    # else:
    
    padding_mode_sift = 'zeros'

        
    # print(f"train: {len(trainloader.dataset)}")
    # train: 60000
    # print(f"trainloader: {trainloader.dataset[0][0].numel()}")
    # trainloader: 784 (28*28 pixels)

    sample = trainloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    # CapLayer(25, 784, 40, 40)
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
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
    fake_transformation = torch.zeros((BATCH_SIZE, LEN_POSE)).to(DEVICE) 
    fake_img = torch.zeros((BATCH_SIZE, IMG_C, IMG_H, IMG_W)).to(DEVICE)
    model_stats = summary(capL, input_data=[fake_img, fake_transformation], verbose=0)
    save_summary_to_file(model_stats, RESULTS_DIR)

    # poses = []
    loss_history = [] # save the loss for each iteration to plot later

    # Initialize dictionaries to store gradient flow data for plotting
#     grad_flow_caps = {} 
#     grad_flow_layers = {
#     'inp_rec': [],
#     'rec_xy': [],
#     'rec_prob': [],
#     'xy_gen': [],
#     'gen_out': []
# }
    stop = 0 
    # dxy = torch.zeros(size=(BATCH_SIZE, LEN_POSE), device=DEVICE, dtype=torch.float32) 
    len_batch_size = len(trainloader) - 2 # To save last Input, Output, Target images of each epoch
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        n_batches = len(trainloader)
        running_loss = torch.zeros((), device=DEVICE)   # acumula na GPU, sem sync
        epoch_loss_tensors = []                          # guarda tensores para o plot, sem sync
        
        for i, (inp, _) in enumerate(trainloader):
            optimizer.zero_grad(set_to_none=True)

            # inp shape: torch.Size([64, 1, 28, 28])
            inp = inp.to(DEVICE, non_blocking=True)
            target, dxy = BatchShift_torch(inp, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], padding_mode_sift, DEVICE, LEN_POSE)
            out = capL(inp, dxy)
            out = out.view(-1, IMG_C, IMG_H, IMG_W)
            loss = crit(out, target)
            if i == len_batch_size: # Save the input, output images for the first
                Save_In_Out_Target_Images(inp, target, out, epoch, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)
            loss.backward()
            optimizer.step()
            
            loss_detached = loss.detach()
            running_loss += loss_detached

            if i % 100 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (i + 1) / elapsed
                eta = (n_batches - i - 1) / batches_per_sec
                print(f"\rEpoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{n_batches} "
                    f"| {batches_per_sec:.2f} batch/s | ETA época: {eta:.1f}s",
                    end="", flush=True)


            # MEAN GRADIENTS FOR EACH CAPSULE
        #    grad_flow_caps = Save_Mean_Gradients_by_capsule(capL, grad_flow_caps)
            # MEAN GRADIENTS FOR EACH LAYER 
        #    grad_flow_layers = Save_Mean_Gradients_by_layer(capL, grad_flow_layers)

        #PlotGenrative(epoch, capL, IMG_C, IMG_H, IMG_W, RESULTS_DIR_GENERATIVE, num_capsule= NUM_CAPS, num_generative=CAP_GEN)
        current_loss = (running_loss / n_batches).item()
        # loss_history.extend(torch.stack(epoch_loss_tensors).cpu().tolist())  # 1 sync p/ toda a época

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(capL.state_dict(), f'{RESULTS_DIR}/best_model.pth')
            stop = 0
        else:
            stop += 1
            if stop >= 5:
                print(f"\nEarly stopping at epoch {epoch+1} due to no improvement in loss for 5 consecutive epochs.")
                break
        
        diff_time = time.time() - start_time
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]; Time: {(diff_time):.2f} seconds; Loss: {current_loss:.4f}") 
        Loss_Txt(epoch, NUM_EPOCHS, diff_time, current_loss, RESULTS_DIR_LOSS)
        
        # Save_Gradients(capL, epoch, RESULTS_DIR_GRADIENTS) # For Future Works 
        #Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS, window = 50)
        #Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS)

        #Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        #Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

        # Plot_Poses(poses, RESULTS_DIR_POSES, epoch)