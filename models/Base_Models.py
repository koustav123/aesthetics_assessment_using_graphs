from torch import nn
from torch_geometric.nn import GlobalAttention
import torch
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class GATTP(nn.Module):
    def __init__(self, in_features =1024, out_features=64, heads=32):
        super(GATTP, self).__init__()
        self.heads = heads
        self.encoder = nn.Linear(in_features, out_features)
        for i in range(self.heads):
            setattr(self, "head%d" % i, GATP_Basic(out_features))

    def forward(self, x, batch):
        x = self.encoder(x)
        x = torch.cat([getattr(self, "head%d" % i)(x, batch) for i in range(self.heads)], dim = 1)
        # return x
        return F.relu(x)

class GATTP_1(nn.Module):
    #No encoder
    def __init__(self, out_features=64, heads=32):
        super(GATTP_1, self).__init__()
        self.heads = heads
        # self.encoder = nn.Linear(in_features, out_features)
        for i in range(self.heads):
            setattr(self, "head%d" % i, GATP_Basic(out_features))

    def forward(self, x, batch):
        # x = self.encoder(x)
        # pdb.set_trace()

        x = torch.stack([getattr(self, "head%d" % i)(x, batch) for i in range(self.heads)]).mean(0)
        # return x
        return F.relu(x)


class GATP_Basic(nn.Module):
    def __init__(self, in_features=64):
        super(GATP_Basic, self).__init__()
        self.net = GlobalAttention(gate_nn=nn.Linear(in_features,1))

    def forward(self, x, batch):
        return self.net(x, batch)