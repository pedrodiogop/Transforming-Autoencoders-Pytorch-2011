from Capsule import Capsule
import torch.nn as nn
import torch
import torch.nn.functional as F



class CapLayer(nn.Module):
    # CapLayer(25, 784, 40, 40)
    def __init__(self, num_caps, in_dim, cap_rec, cap_gen):
        super(CapLayer, self).__init__()
        self.caps = nn.ModuleList([
                # Capsule(784, 40, 40)
                Capsule(in_dim, cap_rec, cap_gen)
                for _ in range(num_caps)])
        # print(f"{len(self.caps)} capsules created") 
        # 25 capsules created

    def forward(self, X, delxy, sep = False):
        caps_out = []
        R = [] # Pose (x,y), each capsule
        Prob = [] # Probability of feature being present, each capsule
        for cap in self.caps:
            result = cap(X, delxy, sp = sep)
            if sep:
                out, r, prob = result
                R.append(r)
                Prob.append(prob)
            else:
                out = result
            caps_out.append(out)
            
        # print(f"{len(caps_out)} capsules processed") # 25 capsules processed
        # print(f"{caps_out[0].size()}") # torch.Size([64, 784]) 
        # print(f"{caps_out[0][0].size()}") # torch.Size([784])

        stacked = torch.stack(caps_out, dim=0)
        # stacked creates a new dimension for them
        # [25, 64, 784] 

        t = torch.sum(stacked, dim=0)
        # print(f"{t.size()}") # torch.Size([64, 784])

        if sep:
            all_poses = torch.stack(R, dim=0)
            # print(f"{all_poses.size()}") # torch.Size([25, 64, 2])
            all_probs = torch.stack(Prob, dim=0)
            # print(f"{all_probs.size()}") # torch.Size([25, 64, 1])
            r = (all_poses * all_probs).sum(dim=0) / all_probs.sum(dim=0)
            # print(f"{r.size()}") # torch.Size([64, 2])

        return t if not sep else (t, r)
        # t -> is the sum of the contributions of all capsules to each pixel in the output image
        # We didnt use the sigmoid activation on t because we will use BCEWithLogitsLoss which 
        # combines a sigmoid layer and the BCELoss in one single class. This is more numerically stable than using a plain Sigmoid followed by a BCELoss as it takes care of the log-sum-exp trick for us.
        # r -> is the weighted average pose (x,y) across capsules