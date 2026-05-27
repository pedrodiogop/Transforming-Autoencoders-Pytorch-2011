import torch.nn as nn
import torch.nn.functional as F
import torch

# Capsule(784, 40, 40)
class Capsule(nn.Module):
    def __init__(self, input_dim, cap_rec, cap_gen, len_pose):
        super(Capsule, self).__init__()
        self.inpdim = input_dim
        self.cap_rec = cap_rec
        self.cap_gen = cap_gen
        self.cap_xy = len_pose
        # 28*28 -> 40 recognition units 
        self.inp_rec = nn.Linear(self.inpdim, self.cap_rec)

        # 40 -> 2; the pose of the feature in the image (x,y)
        self.rec_xy = nn.Linear(self.cap_rec, self.cap_xy)

        # 40 -> 1; probability of the feature being present in the image
        self.rec_prob = nn.Linear(self.cap_rec, 1)

        # 2 -> 40 generation units
        self.xy_gen = nn.Linear(self.cap_xy, self.cap_gen)

        # 40 -> 28*28 reconstruction of the image
        self.gen_out = nn.Linear(self.cap_gen, self.inpdim)
        # Shape(784, 40) 
        # 784 each row is the contribution of all 40 units to one output pixel
        # 40 columns, each column represents the weights of one generative unit.

    # inp/X -> batch of images
    # dxy/delxy -> batch of transformations
    def forward(self, X, delxy, sp = False): 
        # flatten the input images from (B, 1, 28, 28) to (B, 784)
        X = X.flatten(start_dim=1)
        # print(X.size()) 
        # torch.Size([64, 784])

        # print(delxy.size()) 
        # torch.Size([64, 2])

        cap = F.leaky_relu(self.inp_rec(X), negative_slope=0.01) 
        # print('cap', cap.size()) 
        # cap torch.Size([64, 40])

        x_y = self.rec_xy(cap)
        # print('x_y', x_y.size()) 
        # x_y torch.Size([64, 2])

        prb = torch.sigmoid(self.rec_prob(cap))
        # print('prb', prb.size()) 
        # prb torch.Size([64, 1])
        
        # print('x_y + del', (x_y + delxy).size()) 
        # x_y + del torch.Size([64, 2])   
        gen = F.leaky_relu(self.xy_gen(x_y + delxy), negative_slope=0.01)
        # print('gen', gen.size()) 
        # gen torch.Size([64, 40])

        rec = self.gen_out(gen)
        # print('rec',rec.size()) 
        # rec torch.Size([64, 784])

        output = rec * prb # torch.mul(rec, prb)
        return (output, x_y, prb) if sp else output