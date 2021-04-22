import argparse
import os
import pdb
import sys
from termcolor import colored
import torch
from PIL import Image
from torch_geometric.data import Data as Data_G, Batch as Batch_G
import torch.nn.functional as F

sys.path.append('utils');
sys.path.append('models');

import Augmentation as ag
from models import Graph_Models, Inception_ResNet_V2_Base
import Graph_Utils as GU


def compute_width_height(size, lower_dim=5):
    if size[0] > size[1]:
        return int(lower_dim * size[0] / size[1]), lower_dim
    else:
        return lower_dim, int(lower_dim * size[1] / size[0])


def compute_mean_mos_MLSP(x):
    position_tensor = torch.arange(1, 11).float().unsqueeze(0)
    mean = torch.sum(x.float() * position_tensor, dim=1)
    return mean


def parse_opts():
    parser = argparse.ArgumentParser(description='Aesthetics MTL')

    # Input files
    parser.add_argument('--dir', type=str, default='',
                        help='location of the images')

    # Standard  Hyperparameters during training
    parser.add_argument('--aug', type=str, default='MLSP_AUG_FULL_IMAGE',
                        help='data augmentation strategy')

    # Architecture specifications
    parser.add_argument('--model_name', type=str, default="Avg_Pool_FC",
                        help='name of the pre-trained_model')
    parser.add_argument('--start_from', type=str, default=None,
                        help='location of the pre-trained_model')


    args = parser.parse_args()

    return args


args = parse_opts()
print('############ Parameters ##############')
print(colored('\n'.join(sorted([str(i) + ' : ' + str(j) for (i, j) in vars(args).items()])), 'cyan'))
print('######################################')

DAug_Feat = ag.Augmentation(args.aug, 16);
data_transforms = DAug_Feat.applyTransforms();

model_backbone = Inception_ResNet_V2_Base.inceptionresnetv2()
model_backbone = model_backbone.half().cuda().eval()

model_gnn = getattr(Graph_Models, args.model_name)(args)
model_gnn.load_state_dict(torch.load(args.start_from)['model'])
model_gnn = model_gnn.float().cuda().eval()

print(colored('PTM Loaded from: %s' % args.start_from, 'white'))

inputs = os.listdir(args.dir)

with torch.no_grad():
    for i in inputs:
        # Feature Graph Extraction
        img_T = data_transforms['test'](Image.open(os.path.join(args.dir, i)).convert('RGB')).cuda()
        feat, _ = model_backbone(img_T)
        h, w = compute_width_height(feat[-1].size()[-2:])
        feat = torch.cat([F.interpolate(i, size=(h, w), mode='area') for i in feat], dim=1)

        # Score Regression
        graph_input = [GU.construct_graph_13(img, [0.0, 0.875, 0.0, 0.875]) for img in feat]
        graph_input = [Data_G(x=nodes, pos=pos, y=None) for nodes, pos in graph_input]
        graph_input = Batch_G.from_data_list(graph_input)
        graph_input.x = graph_input.x.float().cuda();
        graph_input.batch = graph_input.batch.cuda()
        graph_input.pos = graph_input.pos.cuda().float()
        scores_dist = model_gnn(graph_input, 'test')['A2']

        # MOS computation
        mos = compute_mean_mos_MLSP(scores_dist.mean(dim=0).cpu())
        print("%s: %f" % (i, mos))
