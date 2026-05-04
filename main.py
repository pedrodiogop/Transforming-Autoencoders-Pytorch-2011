import os
from torchinfo import summary
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from image_utils import BatchShift, Show_Batch, BatchShift_torch
from CapLayer import CapLayer
import torch.optim as optim
import torch.nn as nn
import time
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Treino de CapsuleLayer')
    
    parser.add_argument('--device',     type=str,   default='mps',  help='cpu ou mps')
    parser.add_argument('--batch_size', type=int,   default=64,    help='Tamanho do batch')
    parser.add_argument('--epochs',     type=int,   default=75,     help='Número de épocas')
    parser.add_argument('--num_caps',   type=int,   default=25,    help='Número de cápsulas')
    parser.add_argument('--in_dim',     type=int,   default=784,    help='Dimensão de entrada (28*28)')
    parser.add_argument('--cap_rec',    type=int,   default=40,   help='Dimensão reconstrução cápsula')
    parser.add_argument('--gen_dim',    type=int,   default=40,   help='Dimensão geração')
    parser.add_argument('--lr',         type=float, default=0.001,  help='Learning rate')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    #device = "cpu"
    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # Confirma se o Pytorch reconhece a GPU
    # No entanto o processo de enviar os dados para a GPU não é automático 
    # Using mps device; GPU da Apple confirmada!
    device = args.device
    print(f"Using {device} device")
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    NUM_CAPS = args.num_caps
    IN_DIM = args.in_dim
    CAP_REC = args.cap_rec
    GEN_DIM = args.gen_dim
    lr = args.lr
    best_loss = 1
  

    trainset = datasets.MNIST(
    root="tmp",
    train=True,
    download=True,
    transform=ToTensor() # Transforma as imagens em tensores PyTorch, normalizando os valores dos pixels para o intervalo [0, 1], shape (canal, altura, largura).
)

    # trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # testeloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # print(f"Number of batches: {len(trainloader)}") # Number of batches: 469
    ## a varivavel trainloader é um interavel de dados temos de usar iter() para obter um iterador e depois usar next() para obter o próximo lote de dados   
    # dataiter = iter(trainloader) 
    # # O método next() retorna o próximo lote de dados, que consiste em um tensor de imagens e um tensor de rótulos correspondentes. O tamanho do lote é determinado pelo parâmetro batch_size que foi definido ao criar o DataLoader.
    # images, labels = next(dataiter) 
    # #print('Labels: ', labels)
    # #print('Batch shape: ', images.size())
    # show_batch(images)

    # Vamos divir o codigo em:
    # Main.py -> para o processo de treino e teste
    # CapLayer.py -> para a implementação da camada de cápsulas
    # Capsule.py -> para a implementação das cápsulas individuais
    capL = CapLayer(NUM_CAPS, IN_DIM, CAP_REC, GEN_DIM)
    capL = capL.to(device)
    #print(capL)
    # CapLayer(
    # (caps): ModuleList(
    #     (0-49): 50 x Capsule(
    #     (cp): Linear(in_features=784, out_features=10, bias=True)
    #     (xy): Linear(in_features=10, out_features=2, bias=True)
    #     (pr): Linear(in_features=10, out_features=1, bias=True)
    #     (gn): Linear(in_features=2, out_features=50, bias=True)
    #     (rc): Linear(in_features=50, out_features=784, bias=True)
    #     )
    # )
    # )
    crit = nn.BCELoss() # Binary Cross entropy Loss, vai comparar a imagem reconstruida com a imagem original.
    # A comparação vai ser feita pixel a pixel, ou seja, cada pixel da imagem reconstruida vai ser comparado com o pixel correspondente da imagem original.
    optimizer = optim.Adam(capL.parameters(), lr = 0.01)  
    dxy_fake = torch.zeros((BATCH_SIZE, 2)).to(device)
    img_fake = torch.zeros((BATCH_SIZE, 1, 28, 28)).to(device)

# Usa o input_data em vez de input_size
    # summary(capL, input_data=[img_fake, dxy_fake])
    ii = 0
    s_i = [] # para armazenar os índices dos lotes
    s_l = [] # para armazenar os valores de perda correspondentes a cada lote
    log_gradientes = 'data/gradientes_log.txt'
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()  # Marca o início
        runn_loss = 0.0
        for i, (inp, _) in enumerate(trainloader): # 469 batches por época
            #print(i, end='\r') # 0 1 2 3 4 ... 468
            # print(f"{inp.shape} {inp.dtype}") 16, 1, 28, 28 torch.float32
            # print(f"{inp.numpy().shape} {inp.numpy().dtype}") (16, 1, 28, 28) float32
            inp = inp.to(device)
            target, dxy = BatchShift_torch(inp, [-4, 4], device)   # tudo na GPU
            #target_np, dxy = BatchShift(inp.numpy().copy(), [-4,4])
            #target = torch.from_numpy(target_np).float().view(-1, 1, 28, 28).to(device)
            # target -> imagens deslocadas 
            # dxy -> deslocamentos aplicados a cada imagem do lote
            #dxy = torch.from_numpy(dxy).float().to(device)


            optimizer.zero_grad()
            out = None
            R = None
            # inp -> batch of images
            # dxy -> batch of transformations
            if i % 500 == 0:
                out, R = capL(inp, dxy, sep = True)
            else:
                out = capL(inp, dxy)
            # out -> batch of reconstructed images [16, 784]
            # R -> batch of poses (x,y) [16, 2]
            out = out.view(-1, 1, 28, 28) # reshape the output to match the original image shape
            loss = crit(out, target)
            loss.backward()
            # No teu loop de treino, após loss.backward():

            # with open(log_gradientes, 'a') as f:
            #     f.write(f"--- Epoch: {epoch+1}, Iter: {i+1} ---\n")
            #     for name, param in capL.named_parameters():
            #         if param.grad is not None:
            #             # Calculamos os valores
            #             mean_grad = param.grad.mean().item()
            #             max_grad = param.grad.max().item()
                        
            #             # Escrevemos no ficheiro
            #             linha = f"Camada: {name:.<30} | Grad Médio: {mean_grad:>10.8f} | Max: {max_grad:>10.8f}\n"
            #             f.write(linha)
            #     f.write("-" * 50 + "\n")

            optimizer.step()
            current_loss = loss.item()
            # Dentro do loop de épocas, se a loss atual for a menor:
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.clone() for k, v in capL.state_dict().items()}
            runn_loss += current_loss
            s_i.append(ii)
            s_l.append(loss.item())
            ii += 1
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
        end_time = time.time()    # Marca o fim
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
            weights = capL.caps[k].rc.weight.data.cpu()  # (784, 50)
            for p in range(10):  # primeiras 10 dimensões gerativas    
                ax = axes[k][p]
                ax.imshow(weights[:, p].view(28, 28), cmap='gray')
                ax.axis('off')
                if k == 0:
                    ax.set_title(f'Gen {p}', fontsize=4)
        plt.tight_layout()
        os.makedirs('Imagens_Pesos_Generativos', exist_ok=True)
        plt.savefig(f'Imagens_Pesos_Generativos/pesos_rc_ep_{epoch+1}.png')
        plt.close(fig)

    torch.save(best_state, "best_checkpoint.pth")
    # 1. Garantir que a pasta 'plots' existe
    os.makedirs('images/plots', exist_ok=True)
    # 2. Criar a figura
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s_i, s_l, label=f'Função de Custo')
    # 4. Estilizar o gráfico
    ax.set_title('Função de Custo')
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Loss (BCELoss)')
    ax.grid(True)
    ax.legend()
    # 5. Guardar a imagem
    caminho_plot = 'images/plots/funcao_custo.png'
    fig.savefig(caminho_plot)    
    # 6. Fechar a figura para libertar memória RAM
    plt.close(fig)