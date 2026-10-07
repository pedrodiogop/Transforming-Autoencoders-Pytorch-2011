from Feature_Map_Capsule import Feature_Map_Capsule 
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



class Feature_Map_CapLayer(nn.Module):
    # Feature_Map_CapLayer(25, 784, 40, 40)
    def __init__(self, num_caps, in_dim, cap_rec, cap_gen, len_pose):
        super(Feature_Map_CapLayer, self).__init__()
        self.caps = nn.ModuleList([
                # Feature_Map_Capsule(784, 40, 40)
                Feature_Map_Capsule(in_dim, cap_rec, cap_gen, len_pose)
                for _ in range(num_caps)])
        # print(f"{len(self.caps)} capsules created") 
        # 25 capsules created

    def forward(self, X, delxy):
        caps_out = []
        Prob = [] # Probability of feature being present, each capsule
        for cap in self.caps:
            out, _, prob, gen = cap(X, delxy)
            caps_out.append(out)
            Prob.append(prob)

        capsule_stacked = torch.stack(caps_out, dim=0)
        # [25, 1, 784] 

        reconstruction_image = torch.sum(capsule_stacked, dim=0)
        # print(f"{reconstruction_image.size()}") # torch.Size([1, 784])

        all_probs = torch.stack(Prob, dim=0)

        return capsule_stacked, reconstruction_image, all_probs, gen
        # DONT FORGET TO USE SIGMOID 