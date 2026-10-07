import os
from torchinfo import summary
import torch
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
from aux_functions import Get_Args_SmallNorb, BatchShift_torch, Plot_Loss, PlotGenrative, Loss_Txt_Small_Norb, set_seed, save_summary_to_file
from aux_gradients import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import torchvision
from Class_Small_Norb_Custom import SmallNORBPairDataset_Custom
import torch.nn.functional as F

# def fg_mask(img, thr):
#     #bg = img[:, :, :1, :1] 
#     bg = img.flatten(1).median(dim=1).values.view(-1, 1, 1, 1) # (B, 1, 1, 1) -> median along H*W
#     return ((img - bg).abs() > thr).float()

# def make_mask(x, target, thr, k):
#     with torch.no_grad():
#         #m = torch.maximum(fg_mask(x, thr), fg_mask(target, thr))
#         # stride 1 e padding k//2 para manter o tamanho da máscara igual ao da imagem
#         m = fg_mask(target, thr)
#         m = F.max_pool2d(m, k, stride=1, padding=k // 2) # o k controla a dilatação da máscara, para cobrir melhor o objeto. Tem de ser impar para a a mascara ter o mesmo tamanho da imagem.
#     return m

# def mse_fg_bg(out, target, m, eps=1e-8):
#     se = (out - target) ** 2
#     mse_fg = (se * m).sum() / (m.sum() + eps)
#     mse_bg = (se * (1 - m)).sum() / ((1 - m).sum() + eps)
#     return mse_fg, mse_bg

# def loss_fn(out, target, x, w_fg, thr, k):
#     m = make_mask(x, target, thr, k) # (n, 1, H, W), valores 0/1
#     mse_fg, mse_bg = mse_fg_bg(out, target, m)
#     loss = w_fg * mse_fg + (1 - w_fg) * mse_bg
#     return loss, mse_fg, mse_bg

def Save_In_Out_Target_Images(inp, target, out, epoch, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
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
    
    caminho = os.path.join(RESULTS_DIR_IN_OUT_TARGET_IMAGES, f'Epoch_{epoch:03d}.png')
    plt.imsave(caminho, img)


@torch.no_grad()
def evaluate(model, loader, device, img_shape, crit, RESULTS_DIR_IN_OUT_TARGET_IMAGES_VALIDATION):
    """Loss média no conjunto de teste (sem gradientes)."""
    model.eval()
    running_validation_loss = torch.zeros((), device=DEVICE)
    n_batches = len(loader)
    for i, (x, target, transf) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        transf = transf.to(device, non_blocking=True)

        out = model(x, transf).view(-1, *img_shape)
        running_validation_loss += crit(out, target)
        if i == n_batches - 2: 
            Save_In_Out_Target_Images(x, target, out, epoch, RESULTS_DIR_IN_OUT_TARGET_IMAGES_VALIDATION, DATASET)
    model.train()
    return (running_validation_loss / n_batches).item()

if __name__ == '__main__':
    args = Get_Args_SmallNorb()

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
    PATH_DATASET_NORB = args.dataset_path
    IMAGE_SIZE = args.img_size
    # ITERS_PER_EPOCH = args.iter_per_epoch     # nº de batches por "epoch"
    EVAL_BATCHES = args.eval_batches         # nº de batches no teste a cada epoch
    CROP = 80
    # P_LOSS_FG = args.p_loss_fuct_fg
    # THR_IMAGE_OBJECT = args.thr
    # K_KERNEL = args.k_kernel

    PATIENCE = 20
    
    print(DEVICE)
    lr = args.lr
    set_seed(SEED)

    # Define the directory to save results
    # RESULTS_DIR_POSES = f'{RESULTS_DIR_TRAIN}/Poses'
    # RESULTS_DIR_GRADIENTS = f'{RESULTS_DIR_TRAIN}/Gradients_log.txt'

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{EVAL_BATCHES}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{IMAGE_SIZE}_{CROP}_{lr}_{SEED}'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    #RESULTS_DIR = f'{RESULTS}/MASK/{P_LOSS_FG}_{THR_IMAGE_OBJECT}_{K_KERNEL}'
    RESULTS_DIR_TRAIN = f'{RESULTS_DIR}/Train'
    RESULTS_DIR_VALIDATION = f'{RESULTS_DIR}/Validation'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR_TRAIN}/Loss_Image_TXT'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES = f'{RESULTS_DIR_TRAIN}/In_Out_Target_Images'
    RESULTS_DIR_IN_OUT_TARGET_IMAGES_VALIDATION = f'{RESULTS_DIR_VALIDATION}/In_Out_Target_Images'
    #RESULTS_DIR_GRADIENTS_MEAN_CAPSULES = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Capsule'
    #RESULTS_DIR_GRADIENTS_MEAN_LAYERS = f'{RESULTS_DIR_TRAIN}/Mean_Gradients_by_Layer'
    #RESULTS_DIR_GENERATIVE = f'{RESULTS_DIR_TRAIN}/Generative_Plot'

    trainset = SmallNORBPairDataset_Custom(
        PATH_DATASET_NORB,
        split='train',
        image_size=IMAGE_SIZE,
        crop_size=CROP,
        elev_idx=[0, 1, 2],
        azim_idx=[0, 2, 4],
        light_idx=[2, 4, 5]
    )
    
    testset  = SmallNORBPairDataset_Custom(
        PATH_DATASET_NORB, 
        split='test',  
        image_size=IMAGE_SIZE,
        crop_size=CROP,
        elev_idx=[0, 1, 2],
        azim_idx=[0, 2, 4],
        light_idx=[2, 4, 5]
    )

    # print(len(trainset)) # train_dataset: 47239200 || 18225
    
    # train_sampler = RandomSampler(trainset, replacement=True, num_samples=ITERS_PER_EPOCH * BATCH_SIZE)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True, persistent_workers=True)
    print(len(trainloader)) # trainloader: 10000 || 225

    # trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    def make_test_loader():

        g = torch.Generator().manual_seed(SEED)

        sampler = RandomSampler(testset, replacement=False,
                                num_samples=EVAL_BATCHES * BATCH_SIZE, generator=g)

        return DataLoader(testset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)

    # sample, _, _, _, _, _ = trainloader.dataset[0]  # (C, H, W)
    sample, _, _= trainloader.dataset[0]  # (C, H, W)
    # print(sample.shape)  # (1, 32, 32)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL = capL.to(DEVICE)
    # capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE).to(DEVICE)
    # capL.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    crit = nn.MSELoss() # nn.BCEWithLogitsLoss() # 
    optimizer = optim.Adam(capL.parameters(), lr)  

    # To check the model architecture 
    fake_transformation = torch.zeros((BATCH_SIZE, LEN_POSE)).to(DEVICE) 
    fake_img = torch.zeros((BATCH_SIZE, IMG_C, IMG_H, IMG_W)).to(DEVICE)
    model_stats = summary(capL, input_data=[fake_img, fake_transformation], verbose=0)
    save_summary_to_file(model_stats, RESULTS_DIR)

    best_test_validation_loss = float('inf') 
    stop = 0 
    n_batches = len(trainloader) # To save last Input, Output, Target images of each epoch
    # os.makedirs(RESULTS_DIR, exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        running_loss = torch.zeros((), device=DEVICE)
        capL.train()
        for i, (x, target, tansf) in enumerate(trainloader):
            x = x.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            tansf = tansf.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = capL(x, tansf).view(-1, IMG_C, IMG_H, IMG_W)
            loss = crit(out, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
            if i == n_batches - 2:
                Save_In_Out_Target_Images(x, target, out, epoch, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET)
            if i % 100 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (i + 1) / elapsed
                eta = (n_batches - i - 1) / batches_per_sec
                print(f"\rEpoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{n_batches} "
                    f"| {batches_per_sec:.2f} batch/s | ETA época: {eta:.1f}s",
                    end="", flush=True)
        train_loss = (running_loss / n_batches).item()
        validation_loss = evaluate(capL, make_test_loader(), DEVICE, (IMG_C, IMG_H, IMG_W), crit, RESULTS_DIR_IN_OUT_TARGET_IMAGES_VALIDATION)

        # ---------- checkpoint + early stopping (sobre a loss de teste) ----------
        if validation_loss < best_test_validation_loss:
            best_test_validation_loss = validation_loss
            torch.save(capL.state_dict(), f'{RESULTS_DIR}/best_model.pth')
            stop = 0
        else:
            stop += 1

        diff_time = time.time() - start_time
        text = f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]; Time: {diff_time:.2f}s; Train loss: {train_loss:.5f}; Validation loss: {validation_loss:.5f}; Best: {best_test_validation_loss:.5f}; Patience: {stop}/{PATIENCE}\n"
        
        print(text)
        Loss_Txt_Small_Norb(text, RESULTS_DIR_LOSS)

        if stop >= PATIENCE:
            print(f"Early stopping na epoch {epoch+1}: sem melhoria na loss de teste "
                  f"durante {PATIENCE} epochs consecutivas.")
            break
        

        # Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        # Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

        # Plot_Poses(poses, RESULTS_DIR_POSES, epoch)