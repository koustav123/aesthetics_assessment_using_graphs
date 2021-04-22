#System Modules
import torch
import sys
from tqdm import tqdm
from termcolor import colored
import os
import h5py
import pdb
import torch.nn.functional as F

#Custom Modules
sys.path.append('utils');
sys.path.append('dataloaders');
sys.path.append('models');
import opts_extractor as opts
import Augmentation as ag
import Dataloader_misc as dlm
import Model_misc as mdm


args = opts.parse_opts()
model = mdm.load(args, range(10))
model.eval()


print('############ Parameters ##############')
print(colored('\n'.join(sorted([str(i) + ' : ' + str(j) for (i, j) in vars(args).items()])), 'cyan'))
print('######################################')

# input augmentation
DAug = ag.Augmentation(args.aug, args.data_precision);
data_transforms = DAug.applyTransforms();
# target augmentation
TAug = ag.Augmentation(args.aug_target, args.data_precision);
target_transforms = TAug.applyTransforms();

# load dataloader
dsets, dset_loaders, dset_sizes, dset_classes = dlm.load_data_loader(args, args.db,\
 data_transforms, target_transforms)

# open a HDF5 file for writing output
feature_file = h5py.File(os.path.join(args.save_feat, args.feature_file_name), 'w')

# name the groups for storing 8 augmented inputs for a single input
data_groups = ['%d_%d_%d' % (i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
_ = [feature_file.create_group(g) for g in data_groups]

# function to resize the smaller dimension to 5
def compute_width_height(size, lower_dim = 5):
    if size[0]>size[1]:
        return (int(lower_dim * size[0]/size[1]), lower_dim)
    else:
        return (lower_dim, int(lower_dim * size[1]/size[0]))

# function where the magic happens
def extract_HDF5():
    pbar_total = tqdm(range(args.num_epochs), position=0, leave=True, unit=' epochs')
    for epoch in pbar_total:
        with torch.no_grad():
            for phase in ['train', 'test']:
                torch.cuda.empty_cache()
                pbar_epoch = tqdm(dset_loaders[phase], position=1, leave=False, unit=' batches')
                for count, data in enumerate(pbar_epoch):
                    if data:
                        inputs, labels, ids = data;
                        # pdb.set_trace()
                        outputs,_ = model(inputs.squeeze(0).cuda())

                        h,w = compute_width_height(outputs[-1].size()[-2:])
                        feat = torch.cat([F.interpolate(i, size=(h,w), mode='area') for i in outputs],
                                      dim=1)
                        for c, g in enumerate(data_groups):
                            dset = feature_file[g].create_dataset(ids[0], feat[c].size(), dtype="float16")
                            dset[...] = feat[c].cpu().numpy()

        feature_file.close()

extract_HDF5()
