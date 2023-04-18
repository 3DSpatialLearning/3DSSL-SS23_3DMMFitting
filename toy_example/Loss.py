import torch
import torch.nn as nn

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
    def forward(self, pc1, pc2):
        distances = torch.norm(pc1 - pc2, dim=2)  # shape: (BS, M)
        return distances.mean()
