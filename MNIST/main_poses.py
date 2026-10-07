from aux_functions import Get_Args, set_seed
from aux_function_poses import BatchShift_torch, Save_In_Out_Target_Images, Save_Plot_Poses_LessOriginal_MoreOriginal
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from CapLayer import CapLayer
import torch
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt



def Save_In_Out_Target_Imagess(inp, target, out, i, RESULTS_DIR_IN_OUT_TARGET_IMAGES, DATASET):
    inp = inp.detach().cpu()
    out = torch.sigmoid(out).detach().cpu() if ('CIFAR' not in DATASET and 'Mine' not in DATASET) else out.clamp(0, 1).detach().cpu()

    if target is not False and target is not None:
        target = target.detach().cpu()
        batch = torch.cat([inp, target, out], dim=3)
    else: 
        batch = torch.cat([inp, out], dim=3)
         
    im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=False, padding=2, pad_value=0.5)
    # To have the real values we need to set normalize=False. 
    # This way the reconstrution image is not manipulated from the original
    # im_tensor = torchvision.utils.make_grid(batch, nrow=8, normalize=True, padding=2, pad_value=0.5) 
    img = np.transpose(im_tensor.numpy(), (1, 2, 0))
    # img = np.clip(img, 0, 1) 
    
    diretorio = f'{RESULTS_DIR_IN_OUT_TARGET_IMAGES}'
    os.makedirs(diretorio, exist_ok=True)
    caminho = os.path.join(diretorio, f'batch_{i:05d}.png')
    plt.imsave(caminho, img)


if __name__ == '__main__':
    args = Get_Args()

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
    lr = args.lr

    print(DEVICE)
    set_seed(SEED)


    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{lr}_{SEED}'
    RESULTS_DIR_TEST_POSES = f'{RESULTS_DIR}/Test/Equivariance'
    RESULTS_DIR_TEST_POSES_COMPARISON_X = f'{RESULTS_DIR_TEST_POSES}/Capsule_Pose_Analysis/X'
    RESULTS_DIR_TEST_POSES_COMPARISON_Y = f'{RESULTS_DIR_TEST_POSES}/Capsule_Pose_Analysis/Y'
    POSES_DIR_IMAGES = f'{RESULTS_DIR_TEST_POSES}/Images_Reconstruction'

    os.makedirs(RESULTS_DIR_TEST_POSES_COMPARISON_X, exist_ok=True)
    os.makedirs(RESULTS_DIR_TEST_POSES_COMPARISON_Y, exist_ok=True)
    os.makedirs(POSES_DIR_IMAGES, exist_ok=True)

    # padding_mode_sift = 'border' if 'CIFAR' in DATASET else 'zeros'  
    padding_mode_sift = 'zeros'  

    dataset_class = getattr(datasets, DATASET)
    trainset = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
    testeloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    sample = testeloader.dataset[0][0]  # (C, H, W)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_test = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_test = capL_test.to(DEVICE)


    capL_test.load_state_dict(torch.load(f"{RESULTS_DIR}/best_model.pth", map_location=DEVICE))
    capL_test.eval()
     
    # Save the data for plot later 
    all_data = {cap_idx: {'x_orig': [], 'y_orig': [], 
                       'x_right': [], 'y_right': [],
                       'x_left': [], 'y_left': [],
                       'prob_right': [], 'prob_left': []} 
            for cap_idx in range(NUM_CAPS)}

    CONSTANT_TRANSLATION_X = 0
    
    with torch.no_grad():
        for i, (img, _) in enumerate(testeloader):
            img = img.to(DEVICE)
            dxy_zeros = torch.zeros(size=(img.size(0), LEN_POSE), device=DEVICE, dtype=torch.float32) # Original Image
            img_cap_dxyzeros, poses_delzeros, all_prob = capL_test(img, dxy_zeros, sep=True)
            # print(poses_delzeros[study_capsule_index].size()) # torch.Size([25, 64, 2])

            # displacement pixel = CONSTANT_TRANSLATION_X
            target_right, dxy_right = BatchShift_torch(img, -CONSTANT_TRANSLATION_X, padding_mode_sift, DEVICE, LEN_POSE)
            target_left, dxy_left = BatchShift_torch(img, CONSTANT_TRANSLATION_X, padding_mode_sift, DEVICE, LEN_POSE)

            # To Plot Imagens 
            out_right  = capL_test(img, dxy_right)
            out_left = capL_test(img, dxy_left)

            if (i == 0 or i == 3 or i == 14 or i == 20 or i == (len(testeloader) - 2)): 
                img_cap_dxyzeros = img_cap_dxyzeros.view(-1, IMG_C, IMG_H, IMG_W)
                # img_cap_dxyzeros = torch.sigmoid(img_cap_dxyzeros).detach().cpu()
            
                out_right = out_right.view(-1, IMG_C, IMG_H, IMG_W)
                # out_right = torch.sigmoid(out_right).detach().cpu()

                out_left = out_left.view(-1, IMG_C, IMG_H, IMG_W)
                # out_left = torch.sigmoid(out_left).detach().cpu()

                # Save_In_Out_Target_Images(img, target_right, target_left, img_cap_dxyzeros, out_right,  out_left, i, POSES_DIR_IMAGES)
                print(f'Img|target_right|target_left|del0|delright|del_left Save in {POSES_DIR_IMAGES}')
                Save_In_Out_Target_Imagess(img, target_right, out_right, i, POSES_DIR_IMAGES, DATASET)


            # # To Plot Poses
            # _ , poses_right, prob_right = capL_test(target_right, dxy_right, sep=True)
            # _ , poses_left, prob_left = capL_test(target_left, dxy_left, sep=True)
            # # poses_right -> Pose target_right Image Before applying dxy_right
            # # poses_left -> Pose target_left Image Before applying dxy_left
            # # poses_delzeros -> Pose Original Image

        #     for cap_idx in range(NUM_CAPS):
        #         all_data[cap_idx]['x_orig'].append(poses_delzeros[cap_idx, :, 0].cpu().numpy())
        #         all_data[cap_idx]['y_orig'].append(poses_delzeros[cap_idx, :, 1].cpu().numpy())
        #         all_data[cap_idx]['x_right'].append(poses_right[cap_idx, :, 0].cpu().numpy())
        #         all_data[cap_idx]['y_right'].append(poses_right[cap_idx, :, 1].cpu().numpy())
        #         all_data[cap_idx]['x_left'].append(poses_left[cap_idx, :, 0].cpu().numpy())
        #         all_data[cap_idx]['y_left'].append(poses_left[cap_idx, :, 1].cpu().numpy())
        #         all_data[cap_idx]['prob_right'].append(prob_right[cap_idx, :, 0].cpu().numpy())
        #         all_data[cap_idx]['prob_left'].append(prob_left[cap_idx, :, 0].cpu().numpy())

        # for cap_idx in range(NUM_CAPS):
        #     d = {k: np.concatenate(v, axis=0) for k, v in all_data[cap_idx].items()}

        #     mask_right = d['prob_right'] >= 0.80
        #     mask_left = d['prob_left'] >= 0.80

        #     poses_combined_x_right = np.stack([d['x_right'], d['x_orig']], axis=1)[mask_right]
        #     poses_combined_x_left = np.stack([d['x_left'], d['x_orig']], axis=1)[mask_left]

        #     poses_combined_y_right = np.stack([d['y_right'], d['y_orig']], axis=1)[mask_right]
        #     poses_combined_y_left = np.stack([d['y_left'], d['y_orig']], axis=1)[mask_left]
            
        #     if len(poses_combined_x_right) < 1 or len(poses_combined_x_left) < 1:
        #         print(f"Capsule {cap_idx} — Insufficient data after filtering (right={len(poses_combined_x_right)}, left={len(poses_combined_x_left)}), skip...")
        #         continue

        #     if len(poses_combined_y_right) < 1 or len(poses_combined_y_left) < 1:
        #         print(f"Capsule {cap_idx} — Insufficient data after filtering (right={len(poses_combined_y_right)}, left={len(poses_combined_y_left)}), skip...")
        #         continue

        #     Save_Plot_Poses_LessOriginal_MoreOriginal(
        #         poses_combined_x_right, poses_combined_x_left, RESULTS_DIR_TEST_POSES_COMPARISON_X, cap_idx, CONSTANT_TRANSLATION_X, "X"
        #     )

        #     Save_Plot_Poses_LessOriginal_MoreOriginal(
        #         poses_combined_y_right, poses_combined_y_left, RESULTS_DIR_TEST_POSES_COMPARISON_Y, cap_idx, CONSTANT_TRANSLATION_X, "Y"
        #     )

        # for threshold in [0.5, 0.6, 0.7, 0.8]:
        #     mask = all_prob_less >= threshold
        #     print(f"threshold={threshold} → {mask.sum()} poses ({mask.mean()*100:.1f}%)")
        #     threshold=0.5 → 8796 poses (88.0%)
        #     threshold=0.6 → 6216 poses (62.2%)
        #     threshold=0.7 → 3433 poses (34.3%)
        #     threshold=0.8 → 1259 poses (12.6%)
        # Aplica o filtro
        # print(f"Total de poses:    {len(poses_combined_less)}") 10000
        # print(f"Poses filtradas:   {poses_combined_less_filtered.shape[0]}") 6216
        # print(f"Poses removidas:   {(~mask).sum()}") 3784












