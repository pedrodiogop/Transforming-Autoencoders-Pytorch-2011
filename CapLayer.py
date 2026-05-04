from Capsule import Capsule
import torch.nn as nn
import torch
import torch.nn.functional as F



class CapLayer(nn.Module):
    def __init__(self, num_caps, in_dim, cap_dim, gen_dim):
        super(CapLayer, self).__init__()
        self.caps = nn.ModuleList([
                Capsule(in_dim, cap_dim, gen_dim)
                for _ in range(num_caps)])
        # print(f"{len(self.caps)} capsules created") 50 capsules created

    def forward(self, X, delxy, sep = False):
        #caps_out = [cap(X, delxy) for cap in self.caps]
        #print(f"{len(caps_out)}") # 50 capsules processed
        # print(f"{caps_out[0].size()}")  torch.Size([16, 784]) 
        # [16,784] sao 16 imagens reconstruidas, cada uma com 784 pixels (28*28)
        caps_out = []
        R = []
        for cap in self.caps:
            if sep:
                out, r = cap(X, delxy, sp = True)
                caps_out.append(out)
                R.append(r)
            else:
                out = cap(X, delxy)
                caps_out.append(out)
        #print(f"{len(caps_out)} capsules processed") # 50 capsules processed
        #print(f"{caps_out[0].size()}") # torch.Size([16, 784]) 
        stacked = torch.stack(caps_out, dim=0)
         # [50,16,784] sao 50 capsules, cada uma processando um lote de 16 imagens e produzindo uma imagem reconstruida de 784 pixels (28*28)
        t = torch.sum(stacked, dim=0)
        #print(f"{t.size()}") # torch.Size([16, 784])
        # [16,784] sao 16 imagens reconstruidas, cada uma com 784
        if sep: 
            all_poses = torch.stack(R, dim=0)
            r = torch.mean(all_poses, dim=0)
        if not sep:
            return torch.sigmoid(t)
        return torch.sigmoid(t), r