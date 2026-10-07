import os
import torch
from torchvision.utils import make_grid, save_image
from Class_Small_Norb_Custom import SmallNORBPairDataset_Custom
from aux_functions import Get_Args_SmallNorb
from CapLayer import CapLayer




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
    #ITERS_PER_EPOCH = args.iter_per_epoch     # nº de batches por "epoch"
    EVAL_BATCHES = args.eval_batches         # nº de batches no teste a cada epoch
    CROP = 80

    RESULTS_DIR = f'Results/{DATASET}/{BATCH_SIZE}_{EVAL_BATCHES}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{IMAGE_SIZE}_{CROP}_{lr}_{SEED}'
    OUT_DIR = f'{RESULTS_DIR}/Test/Latent_Space_Manipulation_Images'

    os.makedirs(OUT_DIR, exist_ok=True)

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

    # print(f"Sample shape: {trainset[0][0].shape}") 18225
    # testloader_latent_space = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    sample, _, _= trainset[18000]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE).to(DEVICE)
    capL_test.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_test.eval()

    x = sample.unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    trans_identidade = torch.zeros((1, LEN_POSE), device=DEVICE)  # (1, 3)
    transf = torch.tensor([[0.4, 0.0, -0.2]], device=DEVICE)  # (1, 3)
    with torch.no_grad():
        identidade = capL_test(x.view(1, -1), trans_identidade).view(1, IMG_C, IMG_H, IMG_W)
        out = capL_test(x.view(1, -1), transf).view(1, IMG_C, IMG_H, IMG_W)
    identidade = identidade.clamp(0, 1)
    out = out.clamp(0, 1)

    grid = make_grid(torch.cat([x, identidade, out], dim=0).cpu(), nrow=3, padding=2, pad_value=0.5)
    save_image(grid, f'{OUT_DIR}/img.png')
    print(f'Guardado em {OUT_DIR}/img.png')