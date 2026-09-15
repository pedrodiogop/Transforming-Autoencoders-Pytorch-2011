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
        self.cap_pose = len_pose
        # 28*28 -> 40 recognition units 
        self.inp_rec = nn.Linear(self.inpdim, self.cap_rec)

        # 40 -> 2; the pose of the feature in the image (x,y)
        self.rec_pose = nn.Linear(self.cap_rec, self.cap_pose)

        # 40 -> 1; probability of the feature being present in the image
        self.rec_prob = nn.Linear(self.cap_rec, 1)

        # 2 -> 40 generation units
        self.pose_gen = nn.Linear(self.cap_pose, self.cap_gen)

        # 40 -> 28*28 reconstruction of the image
        self.gen_out = nn.Linear(self.cap_gen, self.inpdim)
        # Shape(784, 40) 
        # 784 each row is the contribution of all 40 units to one output pixel
        # 40 columns, each column represents the weights of one generative unit.

    # inp/X -> batch of images
    # dxy/delxy -> batch of transformations
    def forward(self, X, transformation, sp = False): 
        # flatten the input images from (B, 1, 28, 28) to (B, 784)
        X = X.flatten(start_dim=1)
        # print(X.size()) 
        # torch.Size([64, 784])

        # print(delxy.size()) 
        # torch.Size([64, 2])

        cap = F.leaky_relu(self.inp_rec(X), negative_slope=0.01) 
        # print('cap', cap.size()) 
        # cap torch.Size([64, 40])

        pose = self.rec_pose(cap)
        # print('x_y', x_y.size()) 
        # x_y torch.Size([64, 2])

        prb = torch.sigmoid(self.rec_prob(cap))
        # print('prb', prb.size()) 
        # prb torch.Size([64, 1])

        # Normalizar a pose para respeitar a regra: cos2+sin2​=1 
        pose_trans = pose[:, 0:2]
        pose_rot = F.normalize(pose[:, 2:4], p=2, dim=1) # porque estamos a nomralizar?
        pose_scale = pose[:, 4]
        pose_shear_x = pose[:, 5]
        pose_shear_y = pose[:, 6]
        normalize_pose = torch.cat([pose_trans, pose_rot], dim=1)
        # R[:,0] = dx_norm R[:,1] = dy_norm R[:,2] = cos_theta R[:,3] = sin_theta
        dx = normalize_pose[:, 0] + transformation[:, 0]
        dy = normalize_pose[:, 1] + transformation[:, 1]
        cos_t = normalize_pose[:, 2] * transformation[:, 2] - normalize_pose[:, 3] * transformation[:, 3]
        sin_t = normalize_pose[:, 3] * transformation[:, 2] + normalize_pose[:, 2] * transformation[:, 3]
        scale = pose_scale + (transformation[:, 4] - 1.0) 
        shear_x = pose_shear_x + transformation[:, 5]
        shear_y = pose_shear_y + transformation[:, 6]

        transformer_pose = torch.stack([dx, dy, cos_t, sin_t, scale, shear_x, shear_y], dim=1)
        

        # print('x_y + del', (x_y + delxy).size()) 
        # x_y + del torch.Size([64, 2])   
        gen = F.leaky_relu(self.pose_gen(transformer_pose), negative_slope=0.01)
        # print('gen', gen.size()) 
        # gen torch.Size([64, 40])

        rec = self.gen_out(gen)
        # print('rec',rec.size()) 
        # rec torch.Size([64, 784])

        output = rec * prb # torch.mul(rec, prb)
        return (output, normalize_pose, prb) if sp else output