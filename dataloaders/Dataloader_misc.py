import sys
import torch
import Custom_Dataset
# import pdb
# from torch.utils.data import WeightedRandomSampler
# from torch_geometric.data import DataLoader as Dataloader_G
# import numpy as np
# import multiprocessing
sys.path.append('../utils');




    
def load_data_loader(args, db, data_transforms, target_transforms):
    batch_sizes = {'train': args.batch_size, 'test': args.batch_size_test}

    if args.id in ['AIAG_Extraction']:
        #pdb.set_trace()
        dsets = {x: Custom_Dataset.AIAG_Dataset(x, args, \
            transform = data_transforms[x]) for x in ['train', 'test']}

    elif args.id in ['AIAG']:
        #pdb.set_trace()
        dsets = {x: Custom_Dataset.AIAG_Dataset_PyTorch_HDF5_MLSP_3(x, args, \
                                                  transform=data_transforms[x]) for x in ['train', 'test']}

    else:
        print('Task ID is wrong or not specified in dataloader: %s'%(args.id))
        exit()


    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], \
    batch_size = batch_sizes[x], shuffle = x == 'train', num_workers = args.n_workers, \
    collate_fn = dsets[x].collate) for x in ['train', 'test']}
        #note how dataloader is configured for shufle. no shuffle during validation

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
    print ('Training Images : %d, Validation Images : %d'\
        %(dset_sizes['train'], dset_sizes['test']))
    #pdb.set_trace()
    return dsets, dset_loaders, dset_sizes, dsets['train'].classes