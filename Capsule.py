import torch.nn as nn
import torch.nn.functional as F
import torch

# Capsule(28*28, 10, 50)
class Capsule(nn.Module):
    def __init__(self, input_dim, cap_dim, gen_dim, xy_dim = 2):
        super(Capsule, self).__init__()
        self.indim = input_dim
        self.cpdim = cap_dim
        self.gndim = gen_dim
        self.xytrn = xy_dim
        # São criados 5 camadas lineares independetes
        # No metodo forward definimos o flow de dados entre as camadas, ou seja, quais camadas recebem a saída de outras camadas como entrada
        # 28*28 -> 10 recognition units 
        self.cp = nn.Linear(self.indim, self.cpdim) #Recognizer units
        # 10 -> 2 resumir a caracteristica da imagem em 2 numeros (x,y)
        self.xy = nn.Linear(self.cpdim, self.xytrn) #estimates of the X and Y
        # 10 -> 1 probabilidade de existir a caracteristica na imagem
        self.pr = nn.Linear(self.cpdim, 1)          #prob of feature
        # 2 -> 50 gerar uma nova imagem a partir dos 2 numeros (x,y)
        self.gn = nn.Linear(self.xytrn, self.gndim) #The generator
        # 50 -> 28*28 reconstruir a imagem a partir dos 50 numeros gerados
        self.rc = nn.Linear(self.gndim, self.indim) #The reconstructed image

    # inp/X -> batch of images
# dxy/delxy -> batch of transformations
    def forward(self, X, delxy, sp = False): 
        X = X.view(-1, 28*28) # flatten the input images, ou seja, transformar cada imagem de 28x28 pixels em um vetor de 784 elementos (28*28 = 784). O -1 é usado para indicar que o número de linhas deve ser inferido automaticamente com base no tamanho do lote. Assim, se o lote tiver 16 imagens, a saída será um tensor de forma (16, 784).
        # print(X.size(), delxy.size()) 
        # torch.Size([16, 784]) torch.Size([16, 2])
        cap = F.relu(self.cp(X))  
        #print('cap', cap.size()) cap torch.Size([16, 10])
        x_y = self.xy(cap)
        # print('x_y', x_y.size()) x_y torch.Size([16, 2])
        prb = torch.sigmoid(self.pr(cap))
        # print('prb', prb.size()) prb torch.Size([16, 1])
        # print('x_y + del', (x_y + delxy).size()) x_y + del torch.Size([16, 2])
        gen = F.relu(self.gn(x_y + delxy)) 
        # print('gen', gen.size()) gen torch.Size([16, 50])
        rec = self.rc(gen)
        # print('rec',rec.size()) rec torch.Size([16, 784])
        if sp:
            return torch.mul(rec,prb), x_y
        else:
            return torch.mul(rec,prb)
        