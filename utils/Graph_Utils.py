import torch
import pdb
import torch.nn.functional as F
import itertools
from random import shuffle
import numpy as np
import time
import torch_geometric.transforms as transforms_G
from math import pi as PI
from torch_geometric.nn import knn_graph, radius_graph




def construct_graph_13(inputs, coordinates = None):
    #pdb.set_trace()
    _,h,w = inputs.size()
    # if h <=1 or w <=1:
    #     #pdb.set_trace()
    #     inputs = F.interpolate(inputs.unsqueeze(0), size=[2, 2], mode='area').squeeze(0)
    #     _, h, w = inputs.size()

    pos = torch.stack([torch.Tensor(k) for k in [*itertools.product(range(h), range(w))]]).int()
    nodes = torch.stack([inputs[:,i,j] for (i,j) in pos])
    return nodes, pos

def compute_width_height(size, lower_dim = 7):
    if size[0]>size[1]:
        return (int(lower_dim * size[0]/size[1]), lower_dim)
    else:
        return (lower_dim, int(lower_dim * size[1]/size[0]))

def construct_graph_14(inputs, coordinates = None):
    #resize feature maps from 10x
    #pdb.set_trace()
    _,h,w = inputs.size()
    #new_h, new_w = compute_width_height((h,w))
    #inputs = F.interpolate(inputs.unsqueeze(0), size=[new_h, new_w], mode='area').squeeze(0)
    #h, w = new_h, new_w

    # if h <=1 or w <=1:
    #     #pdb.set_trace()
    #     inputs = F.interpolate(inputs.unsqueeze(0), size=[2, 2], mode='area').squeeze(0)
    #     _, h, w = inputs.size()

    pos = torch.stack([torch.Tensor(k) for k in [*itertools.product(range(h), range(w))]]).int()
    nodes = torch.stack([inputs[:,i,j] for (i,j) in pos])
    return nodes, pos

def graph_transform_1():
   return transforms_G.Compose([
       transforms_G.Distance(norm=False, cat=True),
       transforms_G.Cartesian(norm=False, cat=True),
       transforms_G.Polar(norm=False, cat=True)
   ])
#
#
def graph_transform_2():
   return transforms_G.Compose([
       transforms_G.Distance(norm=False, cat=True),
       transforms_G.Cartesian(norm=False, cat=True),
       BiPolar(norm=False, cat=True)
   ])

def graph_transform_3():
   return transforms_G.Compose([
       #transforms_G.KNNGraph(k=5),
       #transforms_G.RadiusGraph(0.1),
       #transforms_G.Polar(cat=False)
       Radius_With_Global_Node(),
       # Only_Global_Node()
       # transforms_G.Cartesian(cat=False)
       #transforms_G.Spherical(cat=False)
       #transforms_G.Distance(norm=False, cat=True)
       #BiPolar(norm=False, cat=True)
   ])
def graph_transform_4():
    return transforms_G.Compose([
        # transforms_G.KNNGraph(k=6)
        transforms_G.RadiusGraph(r=4)
    ])
def graph_transform_5():
    return transforms_G.Compose([
        Add_Coordinates()

    ])
class Radius_With_Global_Node(object):
    def __init__(self, r = 4):
        self.radius = r

    def __call__(self, data):
        #pdb.set_trace()
        nodes = data.x
        #global_node = torch.ones([1, data.x.size()[1]])
        global_node = nodes.mean(dim = 0).unsqueeze(0)

        data.x = torch.cat([global_node, nodes])
        pos = data.pos
        #pdb.set_trace()
        edge_index = radius_graph(pos, r = self.radius)
        zero_index =  torch.stack((torch.arange(data.x.size()[0]), torch.zeros(data.x.size()[0]).long()))[:,1:]
        edge_index = torch.cat((zero_index, edge_index+1), dim = 1)
        data.pos = torch.cat((torch.zeros(1, 2).int(), pos+1))
        data.edge_index = edge_index
        return data

class Radius_With_Global_Node_Bi_Index(object):
    def __init__(self, r = 4):
        self.radius = r

    def __call__(self, data):
        #pdb.set_trace()
        nodes = data.x
        #global_node = torch.ones([1, data.x.size()[1]])
        global_node = nodes.mean(dim = 0).unsqueeze(0)

        data.x = torch.cat([global_node, nodes])
        pos = data.pos
        #pdb.set_trace()
        edge_index_local = radius_graph(pos, r = self.radius) + 1
        edge_index_global =  torch.stack((torch.arange(data.x.size()[0]), torch.zeros(data.x.size()[0]).long()))[:,1:]
        #edge_index = torch.cat((zero_index, edge_index+1), dim = 1)
        data.pos = torch.cat((torch.zeros(1, 2).int(), pos+1))
        data.edge_index_local = edge_index_local
        data.edge_index_global = edge_index_global
        return data

class Only_Global_Node(object):
    def __init__(self, r = 4):
        self.radius = r

    def __call__(self, data):
        #pdb.set_trace()
        nodes = data.x
        #global_node = torch.ones([1, data.x.size()[1]])
        global_node = nodes.mean(dim = 0).unsqueeze(0)

        data.x = torch.cat([global_node, nodes])
        pos = data.pos
        #pdb.set_trace()
        #~~edge_index = radius_graph(pos, r = self.radius)
        zero_index  = torch.stack((torch.arange(data.x.size()[0]), torch.zeros(data.x.size()[0]).long()))[:,1:]
        edge_index = zero_index
        data.pos = torch.cat((torch.zeros(1, 2).int(), pos+1))
        data.edge_index = edge_index
        return data

class Zero_Node_As_Global_Node(object):
    # def __init__(self):
    #     # self.radius = r

    def __call__(self, data):
        #pdb.set_trace()
        # nodes = data.x
        #global_node = torch.ones([1, data.x.size()[1]])
        # global_node = nodes.mean(dim = 0).unsqueeze(0)

        # data.x = torch.cat([global_node, nodes])
        pos = data.pos
        #pdb.set_trace()
        #~~edge_index = radius_graph(pos, r = self.radius)
        zero_index  = torch.stack((torch.arange(data.x.size()[0]), torch.zeros(data.x.size()[0]).long()))[:,1:]
        edge_index = zero_index
        # data.pos = torch.cat((torch.zeros(1, 2).int(), pos+1))
        data.edge_index = edge_index
        return data

class KNN_With_Global_Node_1(object):
    def __init__(self, k = 0):
        self.neighb = k

    def __call__(self, data):
        #pdb.set_trace()
        nodes, pos = data.x , data.pos
        pos = pos + 1
        #mean_node = nodes.mean(dim=0).unsqueeze(0)
        mean_node = torch.ones(nodes.size(1)).unsqueeze(0)
        data.x = torch.cat([mean_node, nodes], dim = 0)

        #edge_index = knn_graph(pos, k = self.neighb)
        #edge_index = radius_graph(pos, r = 1)
        zero_index = torch.stack((torch.arange(data.x.size()[0]), torch.zeros(data.x.size()[0]).long()))[:,1:]
        #edge_index = torch.cat((zero_index, edge_index+1), dim = 1)
        edge_index = zero_index
        data.pos = torch.cat((torch.zeros(1, 2), pos.float()))
        data.edge_index = edge_index


        return data

class Add_Coordinates(object):
    def __init__(self, norm=False, cat=True):
        self.norm = norm
        self.cat = cat

    def __call__(self, data):
        # pdb.set_trace()
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        coords = torch.cat([pos[row],pos[col]], dim = 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, coords.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = coords


        return data