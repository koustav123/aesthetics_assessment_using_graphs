import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GlobalAttention, radius_graph, GraphSizeNorm, GraphNorm, LayerNorm, InstanceNorm, PairNorm, knn_graph
from Base_Models import GATTP, GATTP_1
import pdb
from torchvision import models
import pretrainedmodels
import Inception_ResNet_V2_Base


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Graph_Base(torch.nn.Module):
    def __init__(self):
        super(Graph_Base, self).__init__()

    def set_training_mode(self, phase, epoch):
        if phase == 'train':
            self.train(True)
        else:
            self.eval()


class Graph_Block(nn.Module):

    def __init__(self, inf, outf, heads, do):
        super(Graph_Block, self).__init__()
        self.drop = nn.Dropout(p=do)
        self.gl = GATConv(inf, outf, heads=heads, dropout=0.0)
        self.relu = nn.ReLU()
        self.gn = GraphSizeNorm()
        # self.gn = LayerNorm(outf)

    def forward(self, x, edge_index, batch):
        # pdb.set_trace()
        x = self.drop(x)
        x = self.gl(x, edge_index)
        x = self.relu(x)
        x = self.gn(x, batch)
        # x = self.gn(x)
        return x


class Avg_Pool_FC(Graph_Base):
    def __init__(self, args, in_features=16928):
        super(Avg_Pool_FC, self).__init__()
        self.linear = nn.Linear(in_features, 10)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return {'A2': x}


class Avg_Pool_ED(Graph_Base):
    def __init__(self, args, in_features=16928):
        super(Avg_Pool_ED, self).__init__()
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias=False), nn.ReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.1), nn.Dropout(0.5))
        self.lin2 = nn.Sequential(nn.Linear(2048, 1024, bias=False), nn.ReLU(), \
                                  nn.BatchNorm1d(1024, eps=0.1), nn.Dropout(0.5))
        self.lin3 = nn.Linear(1024, args.A2_D)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        x = self.lin01(x)
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        x = self.lin3(x)
        return {'A2': x}

class GCN_GMP(Graph_Base):
    def __init__(self, args, in_features=16928, hidden=2048, heads=8, out_features=3072, edge_dim=2):
        super(GCN_GMP, self).__init__()
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias=False), nn.ReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.2), nn.Dropout(0.7))

        self.conv1 = GCNConv(2048, 2048)
        self.graph_norm_1 = GraphSizeNorm()
        self.lin2 = nn.Sequential(nn.Dropout(0.7), nn.Linear(2048, 1024, bias=False), nn.ReLU(), \
                                  nn.BatchNorm1d(1024, eps=0.2))
        self.lin3 = nn.Linear(1024, args.A2_D)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        radius = 4
        edge_index = radius_graph(pseudo.float(), r=radius, batch=batch)
        x = self.lin01(x)
        x_g1 = F.relu(self.conv1(x, edge_index))
        x = self.graph_norm_1(x_g1, batch)
        # pdb.set_trace()
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        x = self.lin3(x)
        # pdb.set_trace()
        return {'A2': x}

class GAT_x1_GMP(Graph_Base):
    def __init__(self, args, in_features=16928, hidden=2048, heads=8, out_features=3072, edge_dim=2):
        super(GAT_x1_GMP, self).__init__()
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias=False), nn.ReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.2), nn.Dropout(0.7))

        self.conv1 = GATConv(2048, 128, heads=16, dropout=0.0)
        # self.conv2 = GATConv(2048, 64, heads=8, dropout=0.0)
        self.graph_norm_1 = GraphSizeNorm()
        # self.pool = GATTP(in_features=512, out_features=256, heads=8)
        self.lin2 = nn.Sequential(nn.Dropout(0.7), nn.Linear(2048, 1024, bias=False), nn.ReLU(), \
                                  nn.BatchNorm1d(1024, eps=0.2))
        self.lin3 = nn.Linear(1024, args.A2_D)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        # radius = np.random.randint(1,4) if phase == 'train' else 4
        radius = 4
        edge_index = radius_graph(pseudo.float(), r=radius, batch=batch)
        x = self.lin01(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.graph_norm_1(x, batch)
        # pdb.set_trace()
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        x = self.lin3(x)
        # pdb.set_trace()
        return {'A2': x}

class GAT_x1_GATP(Graph_Base):
    def __init__(self, args, in_features=16928, hidden=2048, heads=8, out_features=3072, edge_dim=2):
        super(GAT_x1_GATP, self).__init__()
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias=False), nn.ReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.2), nn.Dropout(0.7))

        self.conv1 = GATConv(2048, 128, heads=16, dropout=0.0)
        self.graph_norm_1 = GraphSizeNorm()
        self.pool = GlobalAttention(gate_nn=nn.Linear(2048, 1))
        self.lin2 = nn.Sequential(nn.Dropout(0.7), nn.Linear(2048, 1024, bias=False), nn.ReLU(), \
                                  nn.BatchNorm1d(1024, eps=0.2))
        self.lin3 = nn.Linear(1024, args.A2_D)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        # radius = np.random.randint(1,4) if phase == 'train' else 4
        radius = 4
        edge_index = radius_graph(pseudo.float(), r=radius, batch=batch)
        x = self.lin01(x)
        x_g1 = F.relu(self.conv1(x, edge_index))
        x = self.graph_norm_1(x_g1, batch)
        # pdb.set_trace()
        x = self.pool(x, batch)
        x = self.lin2(x)
        x = self.lin3(x)
        # pdb.set_trace()
        return {'A2': x.squeeze(1)}

class GAT_x3_GATP_MH(Graph_Base):

    def __init__(self, args, in_features=16928, hidden=2048, heads=8, out_features=3072, edge_dim=2, layers = 3):
        super(GAT_x3_GATP_MH, self).__init__()
        self.layers = layers
        self.lin01 = nn.Sequential(nn.Linear(in_features, 2048, bias=False), nn.LeakyReLU(), \
                                   nn.BatchNorm1d(2048, eps=0.2))
        self.gnn = nn.ModuleList([Graph_Block(2048, 128, 16, 0.8) for i in range(self.layers)])
        self.pool = GATTP_1(out_features=2048, heads=16)
        self.lin2 = nn.Sequential(nn.Dropout(0.8), nn.Linear(2048, 1024, bias=False), nn.ReLU(), \
                                  nn.BatchNorm1d(1024, eps=0.2))
        self.lin3 = nn.Linear(1024, args.A2_D)

    def forward(self, data, phase):
        # pdb.set_trace()
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.pos, data.batch
        # radius = np.random.randint(1,4) if phase == 'train' else 4
        radius = 4
        edge_index = radius_graph(pseudo.float(), r=radius, batch=batch)
        # knn = 10
        # edge_index = knn_graph(x.float(), k=knn, batch=batch)
        x = self.lin01(x)
        for i, layer in enumerate(self.gnn):
            x = layer(x, edge_index, batch)
        x = self.pool(x, batch)
        x = self.lin2(x)
        x = self.lin3(x)
        # pdb.set_trace()
        return {'A2': x.squeeze(1)}

class InceptionResnetV2(Graph_Base):
    def __init__(self, args, pt='imagenet'):
        super(InceptionResnetV2, self).__init__()
        # pdb.set_trace()
        self.trunk = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=pt)
        self.cap = nn.Linear(self.trunk.last_linear.in_features, args.A2_D)
        self.trunk.last_linear = Identity()

    def forward(self, x, phase):
        # pdb.set_trace()
        x = self.trunk(x)
        x = self.cap(x)
        return {'A2': x.squeeze(1)}

class InceptionResnetV2Hierarchial(Graph_Base):
    def __init__(self, args, pt='imagenet'):
        super(InceptionResnetV2Hierarchial, self).__init__()
        # pdb.set_trace()
        self.trunk = Inception_ResNet_V2_Base.inceptionresnetv2()
        self.cap = nn.Linear(self.trunk.last_linear.in_features, args.A2_D)
        self.trunk.last_linear = Identity()

    def forward(self, x, phase):
        pdb.set_trace()
        x = self.trunk(x)
        x = self.cap(x)
        return {'A2': x.squeeze(1)}

