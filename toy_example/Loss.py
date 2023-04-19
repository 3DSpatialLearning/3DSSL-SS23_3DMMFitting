import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    def __init__(self, norm=2):
        super(ChamferDistance, self).__init__()
        if not (norm == 1 or norm == 2):
            raise ValueError("Support for 1 or 2 norm.")
        self.norm = norm

    def pairwise_distances(self, x, y):
        dists = torch.cdist(x, y, p=self.norm)  # (N, P1, P2)
        return dists

    def forward(self, x, y):
        N, P1, D = x.shape
        P2 = y.shape[1]

        if y.shape[0] != N or y.shape[2] != D:
            raise ValueError("y does not have the correct shape.")

        dists = self.pairwise_distances(x, y)  # (N, P1, P2)

        cham_x = torch.min(dists, dim=-1).values  # (N, P1)
        cham_y = torch.min(dists, dim=-2).values  # (N, P2)

        cham_x = cham_x.sum(1)  # (N,)
        cham_y = cham_y.sum(1)  # (N,)

        cham_x /= P1
        cham_y /= P2

        cham_dist = cham_x + cham_y

        return cham_dist