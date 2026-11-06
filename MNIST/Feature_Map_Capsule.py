# import torch.nn as nn
# import torch.nn.functional as F
# import torch

# # Capsule(784, 40, 40)
# class Feature_Map_Capsule(nn.Module):
#     def __init__(self, input_dim, cap_rec, cap_gen, len_pose):
#         super(Feature_Map_Capsule, self).__init__()
#         self.inpdim = input_dim
#         self.cap_rec = cap_rec
#         self.cap_gen = cap_gen
#         self.cap_xy = len_pose
#         # 28*28 -> 40 recognition units 
#         self.inp_rec = nn.Linear(self.inpdim, self.cap_rec)

#         # 40 -> 2; the pose of the feature in the image (x,y)
#         self.rec_xy = nn.Linear(self.cap_rec, self.cap_xy)

#         # 40 -> 1; probability of the feature being present in the image
#         self.rec_prob = nn.Linear(self.cap_rec, 1)

#         # 2 -> 40 generation units
#         self.xy_gen = nn.Linear(self.cap_xy, self.cap_gen)

#         # 40 -> 28*28 reconstruction of the image
#         self.gen_out = nn.Linear(self.cap_gen, self.inpdim)
#         # Shape(784, 40) 
#         # 784 each row is the contribution of all 40 units to one output pixel
#         # 40 columns, each column represents the weights of one generative unit.

#     # inp/X -> batch of images
#     # dxy/delxy -> batch of transformations
#     def forward(self, X, delxy): 

#         X = X.flatten(start_dim=1)

#         cap = F.leaky_relu(self.inp_rec(X), negative_slope=0.01)

#         x_y = self.rec_xy(cap)

#         prb = torch.sigmoid(self.rec_prob(cap))

#         gen = F.leaky_relu(self.xy_gen(x_y + delxy), negative_slope=0.01)

#         rec = self.gen_out(gen)

#         output = rec * 100

#         return output, x_y, prb, gen





import torch.nn as nn
import torch.nn.functional as F
import torch

# Capsule(784, 40, 40)
class Feature_Map_Capsule(nn.Module):
    def __init__(self, input_dim, cap_rec, cap_gen, len_pose):
        super(Feature_Map_Capsule, self).__init__()
        self.inpdim = input_dim
        self.cap_rec = cap_rec
        self.cap_gen = cap_gen
        self.cap_xy = len_pose
        # 28*28 -> 40 recognition units 
        self.inp_rec = nn.Linear(self.inpdim, self.cap_rec)

        # 40 -> 2; the pose of the feature in the image (x,y)
        self.rec_pose = nn.Linear(self.cap_rec, self.cap_xy)

        # 40 -> 1; probability of the feature being present in the image
        self.rec_prob = nn.Linear(self.cap_rec, 1)

        # 2 -> 40 generation units
        self.pose_gen = nn.Linear(self.cap_xy, self.cap_gen)

        # 40 -> 28*28 reconstruction of the image
        self.gen_out = nn.Linear(self.cap_gen, self.inpdim)
        # Shape(784, 40) 
        # 784 each row is the contribution of all 40 units to one output pixel
        # 40 columns, each column represents the weights of one generative unit.

    # inp/X -> batch of images
    # dxy/delxy -> batch of transformations
    def forward(self, X, transformation): 

        X = X.flatten(start_dim=1)

        cap = F.leaky_relu(self.inp_rec(X), negative_slope=0.01)
        # cap = F.relu(self.inp_rec(X)) 

        pose = self.rec_pose(cap)

        prb = torch.sigmoid(self.rec_prob(cap))

        dx = pose[:, 0] + transformation[:, 0] # translaçao X
        dy = pose[:, 1] + transformation[:, 1] # translação Y 
        rotation = pose[:, 2] + transformation[:, 2] # Rotacao
        # sin_t = normalize_pose[:, 3] * transformation[:, 2] + normalize_pose[:, 2] * transformation[:, 3]
        #scale = pose[:, 3] + transformation[:, 3] # scale
        scale_x = pose[:, 3] + transformation[:, 3] # scale x
        scale_y = pose[:, 4] + transformation[:, 4] # scale y
        shear_x = pose[:, 5] + transformation[:, 5] # shear x
        shear_y = pose[:, 6] + transformation[:, 6] # shear y

        transformer_pose = torch.stack([dx, dy, rotation, scale_x, scale_y, shear_x, shear_y], dim=1)

        gen = F.leaky_relu(self.pose_gen(transformer_pose), negative_slope=0.01)
        #gen = F.relu(self.pose_gen(transformer_pose))

        rec = self.gen_out(gen)
        # rec = F.leaky_relu(self.gen_out(gen), negative_slope=20)

        output = rec * prb

        return output, pose, prb, gen