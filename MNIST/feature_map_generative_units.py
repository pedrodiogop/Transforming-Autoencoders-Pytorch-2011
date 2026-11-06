import os
import torch
import torchvision
import matplotlib.pyplot as plt
from aux_functions import Get_Args, BatchShift_torch_Rotation, plot_feature_maps_rows, Input_Target_Output_Images, PlotGenerativeFeatureMaps, set_seed
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Feature_Map_CapLayer import Feature_Map_CapLayer
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_pil_image

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
    SCALE_MIN = args.scale_min
    SCALE_MAX = args.scale_max
    SHEAR = args.shear
    
    lr = args.lr
    set_seed(SEED)

    
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{LEN_POSE}_{RANDOM_TRANSLATION}_{ROTATION_ANGLE}_{SCALE_MIN}_{SCALE_MAX}_{SHEAR}_{lr}_{SEED}'
    RESULTS_DIR_PARTIAL_RECONSTRUCTIONS = f'{RESULTS_DIR}/Partial_Reconstructions'

    padding_mode_sift = 'zeros'

    dataset_class = getattr(datasets, DATASET)
    test_set = dataset_class(root="tmp", train=False, download=True, transform=ToTensor())
    testeloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    image_index = 17 # Index of the image to visualize
    # testeloader.dataset[i][j] -> (image,label)
    sample = testeloader.dataset[image_index][0]  # (C, H, W)
    label = testeloader.dataset[image_index][1]
    print(label)
    IN_DIM = sample.numel()  # total number of pixels (C*H*W)
    IMG_C, IMG_H, IMG_W = sample.shape

    capL_feature_map = Feature_Map_CapLayer(NUM_CAPS, IN_DIM, CAP_REC, CAP_GEN, LEN_POSE)
    capL_feature_map = capL_feature_map.to(DEVICE)

    capL_feature_map.load_state_dict(torch.load(f'{RESULTS_DIR}/best_model.pth', map_location=DEVICE))
    capL_feature_map.eval()
    
    with torch.no_grad():
        sample = sample.unsqueeze(0) # Add a batch dimension to the sample
        sample = sample.to(DEVICE)
        target, transformations = BatchShift_torch_Rotation(sample, [-RANDOM_TRANSLATION, RANDOM_TRANSLATION], [-ROTATION_ANGLE, ROTATION_ANGLE], [SCALE_MIN, SCALE_MAX], [-SHEAR, SHEAR], padding_mode_sift, DEVICE, LEN_POSE)

        # RANDOM_TRANSFORMATIONS = f'{RANDOM_TRANSLATION}_{ROTATION_ANGLE}'
        capsule_stacked, reconstruction_image, all_probs, _ = capL_feature_map(sample, transformations)
        # print(capsule_stacked.size()) torch.Size([25, 1, 784])
        reconstruction_image = reconstruction_image.view(-1, IMG_C, IMG_H, IMG_W)

        num_caps = capsule_stacked.shape[0]
        caps_imgs = capsule_stacked.view(num_caps, IMG_C, IMG_H, IMG_W)
        # print(caps_imgs.size()) # torch.Size([25, 1, 28, 28])

        transformations_str = '_'.join(str(round(float(v), 2)) for v in transformations.flatten())
        file_path = f'{RESULTS_DIR_PARTIAL_RECONSTRUCTIONS}/Capsules/{label}/{transformations_str}'

        plot_feature_maps_rows(caps_imgs, sample, reconstruction_image, target, all_probs, file_path)
        Input_Target_Output_Images(sample, target, reconstruction_image, file_path)
        # PlotGenerativeFeatureMaps(capL_feature_map, sample, transformations, IMG_H, IMG_W, file_path, NUM_CAPS, CAP_GEN)

       
       