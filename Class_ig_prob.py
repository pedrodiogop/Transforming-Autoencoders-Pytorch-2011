import torch
import torch.nn as nn

# versão correta — delxy passado no forward, via additional_forward_args
class PresenceProbWrapper(nn.Module):
    def __init__(self, model, capsule_idx):
        super().__init__()
        self.model = model
        self.capsule_idx = capsule_idx

    def forward(self, X, delxy):
        _, _, all_probs = self.model(X, delxy, sep=True)
        return all_probs[self.capsule_idx].squeeze(-1)