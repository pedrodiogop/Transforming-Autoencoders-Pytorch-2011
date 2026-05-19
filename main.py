import os
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aux_functions import Show_Batch, BatchShift_torch, Plot_Loss
from gradients_aux import Plot_Gradient_Flow_by_layer, Plot_Gradient_Flow_by_capsule, Save_Mean_Gradients_by_capsule, Save_Mean_Gradients_by_layer
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Implementation of Transforming-Autoencoders')
    
    parser.add_argument('--device',     type=str,   default='mps',  help='Device to use for training (e.g., "cpu", "cuda", "mps")')
    parser.add_argument('--batch_size', type=int,   default=64,    help='Batch size for training')
    parser.add_argument('--epochs',     type=int,   default=10,     help='Number of epochs to train')
    parser.add_argument('--num_caps',   type=int,   default=25,    help='Number of capsules')
    parser.add_argument('--cap_rec',    type=int,   default=40,   help='Capsule reconstruction dimension')
    parser.add_argument('--cap_gen',    type=int,   default=40,   help='Generation dimension')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    parser.add_argument('--dataset',    type=str,   default='MNIST', help='Dataset to use for training, only accepts "MNIST", "FashionMNIST" or "CIFAR10".')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # If you want to see your GPU or CPU in action, you can use the following code to check if PyTorch recognizes it and to set the device accordingly:
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    NUM_CAPS = args.num_caps
    CAP_REC = args.cap_rec # encode the image
    CAP_GEN = args.cap_gen # decode the image

    lr = args.lr
    best_loss = 1

    # Define the directory to save results
    RESULTS_DIR = f'Results/{args.dataset}/{BATCH_SIZE}_{NUM_CAPS}_{CAP_REC}_{CAP_GEN}_{lr}'
    RESULTS_DIR_LOSS = f'{RESULTS_DIR}/Image_Loss'
    RESULTS_DIR_GRADIENTS = f'{RESULTS_DIR}/Gradients_log.txt'
    RESULTS_DIR_GRADIENTS_MEAN_CAPSULES = f'{RESULTS_DIR}/Mean_Gradients_by_Capsule'
    RESULTS_DIR_GRADIENTS_MEAN_LAYERS = f'{RESULTS_DIR}/Mean_Gradients_by_Layer'


    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f'{RESULTS_DIR_LOSS}', exist_ok=True) # save loss for each epoch

    if args.dataset == 'FashionMNIST':
        trainset = datasets.FashionMNIST(
            root="tmp",
            train=True,
            download=True,
            transform=ToTensor() # Transform the images to PyTorch tensors, normalizing pixel values to the range [0, 1], shape (channel, height, width).
        )
    elif args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(
            root="tmp",
            train=True,
            download=True,
            transform=ToTensor() # Transform the images to PyTorch tensors, normalizing pixel values to the range [0, 1], shape (channel, height, width).
        )
    else:
        trainset = datasets.MNIST(
            root="tmp",
            train=True,
            download=True,
            transform=ToTensor() # Transform the images to PyTorch tensors, normalizing pixel values to the range [0, 1], shape (channel, height, width).
        )

    # num_workers is the number of subprocesses to use for data loading. If num_workers is set to 0, the data will be loaded in the main process. 
    # If num_workers is greater than 0, that many subprocesses will be used to load the data in parallel, which can speed up data loading, especially for large datasets. 
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
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
    crit = nn.BCEWithLogitsLoss() 
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
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()  # Marca o início
        runn_loss = 0.0
        for i, (inp, _) in enumerate(trainloader): # 469 batches por época
            # print(f"inp shape: {inp.shape}, inp dtype: {inp.dtype} inp length: {len(inp)} inp size: {inp.size()}") 
            # inp shape: torch.Size([64, 1, 28, 28]), inp dtype: torch.float32 inp length: 64 inp size: torch.Size([64, 1, 28, 28])

            #print(i, end='\r') # 0 1 2 3 4 ... 468
            # print(f"{inp.shape} {inp.dtype}") 16, 1, 28, 28 torch.float32
            # print(f"{inp.numpy().shape} {inp.numpy().dtype}") (16, 1, 28, 28) float32
            inp = inp.to(DEVICE)
            target, dxy = BatchShift_torch(inp, [-1, 1], DEVICE)   # tudo na GPU
            #target_np, dxy = BatchShift(inp.numpy().copy(), [-4,4])
            #target = torch.from_numpy(target_np).float().view(-1, 1, 28, 28).to(DEVICE)
            # target -> imagens deslocadas 
            # dxy -> deslocamentos aplicados a cada imagem do lote
            #dxy = torch.from_numpy(dxy).float().to(DEVICE)


            optimizer.zero_grad()
            out = None
            R = None
            # inp -> batch of images
            # dxy -> batch of transformations

            # if i == mid: # na metade da época, vamos visualizar as poses médias (R) das cápsulas
            #     out, R = capL(inp, dxy, sep = True)
            #     poses = R.detach().cpu().numpy()

            #     # 2. Extrai as coordenadas x e y
            #     x = poses[:, 0]
            #     y = poses[:, 1]

            #     # 3. Cria o gráfico
            #     plt.figure(figsize=(8, 6))
            #     plt.scatter(x, y, alpha=0.6, edgecolors='w', label='Poses Médias (R)')

            #     # 4. Adiciona referências visuais
            #     plt.axhline(0, color='black', lw=1, ls='--') # Eixo central X
            #     plt.axvline(0, color='black', lw=1, ls='--') # Eixo central Y

            #     plt.title("Distribuição das Poses Médias no Batch")
            #     plt.xlabel("Coordenada X")
            #     plt.ylabel("Coordenada Y")
            #     plt.grid(True, alpha=0.3)
            #     plt.legend()
            #     plt.show()
            # else:

            out = capL(inp, dxy)
            # out -> batch of reconstructed images [16, 784]
            # R -> batch of poses (x,y) [16, 2]
            out = out.view(-1, IMG_C, IMG_H, IMG_W) # reshape the output to match the original image shape
            # out -> batch of reconstructed images [16, 1, 28, 28]
            # target -> batch of shifted images [16, 1, 28, 28]
            loss = crit(out, target)
            loss.backward()


            # MEAN GRADIENTS FOR EACH CAPSULE
            grad_flow_caps = Save_Mean_Gradients_by_capsule(capL, grad_flow_caps)
    
            # MEAN GRADIENTS FOR EACH LAYER 
            grad_flow_layers = Save_Mean_Gradients_by_layer(capL, grad_flow_layers)

            optimizer.step()
            current_loss = loss.item()
            # Dentro do loop de épocas, se a loss atual for a menor:
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.clone() for k, v in capL.state_dict().items()}
            runn_loss += current_loss
            loss_history.append(current_loss)
            # print(f"Epoch: {epoch}, Iter: {i}, current_loss: {current_loss:.4f}")
            # Epoch: 1, Iter: 561, current_loss: 0.6935

            # if i % 500 == 0:
            #     with open('data/instantiate.txt', 'a') as file:
            #         file.write(f"ep_{epoch+1:05d}_img_{i+1:05d}\n")
            #         # Se R for uma lista de tensores (como no teu forward)
            #         # Vamos converter para uma string de números limpa
            #         for idx, pose in enumerate(R):
            #             pose_data = pose[0].detach().cpu().numpy() 
            #             file.write(f"Cap_{idx}: {pose_data} ")
                    
            #         file.write('\n---\n')
                
            #     runn_loss = 0.0 # Não te esqueças de resetar a loss acumulada!

            if i == 0:
                Show_Batch(inp, target, out, epoch, i, title='Batch Results', save=True)
                # show_batch(inp.data, epoch, i ,title = 'Input', save = True)
                # show_batch(out.data, epoch, i, title = 'output', save = True)
                # print('{0:05d}, {1:05d} loss : {2:6.5f}'.format(epoch+1, i+1, runn_loss / 100))
                runn_loss = 0

        end_time = time.time()    
        epoch_duration = end_time - start_time
        print(f"Época [{epoch+1}/{NUM_EPOCHS}] finalizada em {epoch_duration:.2f} segundos, loss: {current_loss:.4f}") 

                # fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                # fig.suptitle(f'Pesos Generativos rc — Época {epoch+1}')
                
                # for j in range(10):  # primeiras 10 cápsulas
                #     # cada cápsula tem o seu próprio rc: shape (784, 50)
                #     weights = capL.caps[j].rc.weight.data.cpu()  # (784, 50)
                    
                #     # média dos pesos de saída — representa o "template" da cápsula
                #     template = weights.mean(dim=1).view(28, 28)
                    
                #     ax = axes[j // 5][j % 5]
                #     ax.imshow(template, cmap='gray')
                #     ax.set_title(f'Cáp {j}')
                #     ax.axis('off')

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
        
        # Save_Gradients(capL, epoch, RESULTS_DIR_GRADIENTS) # For Future Works 
        Plot_Loss(epoch, loss_history, RESULTS_DIR_LOSS, window = 50)
        Plot_Gradient_Flow_by_capsule(grad_flow_caps, epoch, RESULTS_DIR_GRADIENTS_MEAN_CAPSULES)
        Plot_Gradient_Flow_by_layer(grad_flow_layers, epoch, RESULTS_DIR_GRADIENTS_MEAN_LAYERS)
        # grad_flow_caps = {}  # if you want a graph for each epoch, reset the gradients after plotting
        # grad_flow_layers = {'inp_rec': [], 'rec_xy': [], 'rec_prob': [], 'xy_gen': [], 'gen_out': []}

    torch.save(best_state, "best_checkpoint.pth")