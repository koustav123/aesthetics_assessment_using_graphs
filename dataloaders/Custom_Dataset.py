import torch.utils.data as data
import os
import numpy as np
import torch
import h5py
import pandas as pd
from PIL import Image
import itertools
import pdb

import Graph_Utils as GU
from torch_geometric.data import Data as Data_G, Batch as Batch_G

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class AIAG_Dataset_Pandas(data.Dataset):
    def __init__(self, phase, args, transform=None, target_transform=None, loader=default_loader):
        self.data = args.db
        self.root = args.datapath
        self.feature_path = args.feature_path
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = None
        self.class_count = None
        self.pilot = args.pilot
        self.args = args



    def remove_missing_files(self, ids):
        import os
        missing_indices = [c for c, i in enumerate(
            map(lambda x: os.path.exists(os.path.join(self.root, x)), ids['image_name'].values)) if
                           not i]
        #pdb.set_trace()
        return ids.drop([ids.index[i] for i in missing_indices], axis=0)

    def select_K_training_samples(self, ids):#
        #pdb.set_trace()
        train_idxs = ids[ids['set_x'] == 'training']
        # train_idxs = train_idxs[train_idxs['set_y'] != 'test']
        #test_idxs = ids[ids['set_x'] == 'test']
        train_drop_idxs = [*range(len(train_idxs))];
        #shuffle(train_drop_idxs);
        train_drop_idxs = train_drop_idxs[self.pilot:]
        #test_drop_idxs = range(1000, len(test_idxs))
        #pd.concat([ids.drop([train_idxs.index[i] for i in train_drop_idxs]), ava_rating_test_data], ignore_index=True)
        train_drop_ids = train_idxs.drop([train_idxs.index[i] for i in train_drop_idxs])
        #test_drop_ids = train_drop_ids.drop([test_idxs.index[i] for i in test_drop_idxs])
        return train_drop_ids

    def select_K_test_samples(self, ids):
        #pdb.set_trace()
        #train_idxs = ids[ids['set_x'] != 'test']
        test_idxs = ids[ids['set_x'] == 'test']
        # test_idxs = test_idxs[test_idxs['set_x'] != 'validation']
        # test_idxs = test_idxs[test_idxs['set_y'] != 'training']

        #train_drop_idxs = range(self.pilot, len(train_idxs))
        test_drop_idxs = [*range(len(test_idxs))]; #shuffle(test_drop_idxs); #train_drop_idxs = test_drop_idxs[self.pilot:]
        test_drop_idxs = test_drop_idxs[self.pilot:]
        # if self.pilot != -1:
        #     test_drop_idxs = test_drop_idxs[1000:]
        # else:
        #     test_drop_idxs = test_drop_idxs[self.pilot:]

        #pd.concat([ids.drop([train_idxs.index[i] for i in train_drop_idxs]), ava_rating_test_data], ignore_index=True)
        #train_drop_ids = ids.drop([train_idxs.index[i] for i in train_drop_idxs])
        test_drop_ids = test_idxs.drop([test_idxs.index[i] for i in test_drop_idxs])
        return test_drop_ids

    def __len__(self):
        return len(self.imgs)


class AIAG_Dataset_PyTorch_HDF5_MLSP_3(AIAG_Dataset_Pandas):
    # 8-Crop dataloader and patch averaging
    def __init__(self, phase, args, transform=None, target_transform=None):
        #pdb.set_trace()
        super(AIAG_Dataset_PyTorch_HDF5_MLSP_3, self).__init__(phase, args, transform)
        #pdb.set_trace()
        self.imgs, self.classes, self.class_count = self.make_dataset()
        self.groups = ['%d_%d_%d' % (i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
        # self.graph_T = GU.graph_transform_4()

        #pdb.set_trace()

    def make_dataset(self):
        # pdb.set_trace()
        ids = self.remove_missing_files(pd.read_csv(self.data));
        #if self.pilot != -1:
        if self.phase == 'train':
            ids = self.select_K_training_samples(ids)
        else:
            ids = self.select_K_test_samples(ids)
        img_names = ids['image_name'].values
        imgs = [i for i in img_names]
        labels = ids[[str(i) for i in range(1,11)]].values
        normed_labels = labels/ np.abs(labels.sum(axis =1)[:,None])
        img_labels = [*zip(imgs, normed_labels, img_names)]
        class_count = {'A2': self.args.A2_D}
        classes = range(self.args.A2_D)
        return img_labels, classes, class_count

    def __getitem__(self, index, g_index=-448):
        # self.features = h5py.File(self.feature_path + 'feats_' + self.phase + '.h5', 'r')
        # pdb.set_trace()
        with h5py.File(self.feature_path, 'r') as features:
            # img = features[self.imgs[index][0]].value[0]
            if not self.phase == 'test':
                try:
                    img = np.array(features[self.groups[np.random.randint(len(self.groups))]][self.imgs[index][0]])
                except:
                    print('File %s not found' % (self.imgs[index][0]))
                    return None, None, None
                target = self.imgs[index][1]
                img = self.transform(img)
                nodes, pos = GU.construct_graph_13(img, [0.0, 0.875, 0.0, 0.875])
                d = Data_G(x=nodes, pos=pos, y=target)
                id = self.imgs[index][0]
                # img = features[self.imgs[index][0]].value
            else:
                # img = features[self.groups[0]][self.imgs[index][0]].value
                try:
                    img_list = [np.array(features[g][self.imgs[index][0]]) for g in self.groups]
                except:
                    # print('File %s not found'%(self.imgs[index][0]))
                    return None, None, None
                img_t_list = [self.transform(img) for img in img_list]
                target = self.imgs[index][1]
                # target = [self.imgs[index][1]] * len(img_t_list)
                nodes_pos_list = [GU.construct_graph_13(img, [0.0, 0.875, 0.0, 0.875]) for img in img_t_list]
                d = [Data_G(x=nodes, pos=pos, y=target) for nodes, pos in nodes_pos_list]
                target = [target] * len(d)
                id = [self.imgs[index][0]] * len(d)
            return d, target, id

    def __len__(self):
        return len(self.imgs)


    def collate(self, batch):
        # pdb.set_trace()

        batch = list(filter (lambda x:x is not None and x[0] is not None and x[1] is not None, batch))
        #return dataloader.default_collate(batch)

        #data = [item[0] for item in batch]  # just form a list of tensor
        #data = torch.stack(data, dim = 0)
        if not self.phase == 'test':
            # pdb.set_trace()
            graph = Batch_G.from_data_list([item[0] for item in batch])
            # data = None
            #graph.x = graph.x.cuda(); graph.edge_index = graph.edge_index.cuda(); graph.edge_attr = graph.edge_attr.cuda()
            target = [item[1] for item in batch]
            label = tuple([item[2] for item in batch])
            target = tuple(target)
        else:
            # pdb.set_trace()
            graph = Batch_G.from_data_list([*itertools.chain.from_iterable([item[0] for item in batch])])
            # data = None
            target = [*itertools.chain.from_iterable([item[1] for item in batch])]
            label = tuple(itertools.chain.from_iterable([item[2] for item in batch]))
            target = tuple(target)
        return [graph, target, label]


class AIAG_Dataset(AIAG_Dataset_Pandas):
    def __init__(self, phase, args, transform=None, target_transform=None):
        #pdb.set_trace()
        super(AIAG_Dataset, self).__init__(phase, args, transform)
        self.imgs, self.classes, self.class_count = self.make_dataset()

    def make_dataset(self):
        # pdb.set_trace()
        ids = self.remove_missing_files(pd.read_csv(self.data));
        #if self.pilot != -1:
        if self.phase == 'train':
            ids = self.select_K_training_samples(ids)
        else:
            ids = self.select_K_test_samples(ids)
        img_names = ids['image_name'].values
        imgs = [os.path.join(self.root,i) for i in img_names]
        labels = ids[[str(i) for i in range(1,11)]].values
        normed_labels = labels/ np.abs(labels.sum(axis =1)[:,None])
        img_labels = [*zip(imgs, normed_labels, img_names)]
        class_count = {'A2': self.args.A2_D}
        classes = range(self.args.A2_D)
        return img_labels, classes, class_count

    def __getitem__(self, index):
        # pdb.set_trace()
        path, target, ids = self.imgs[index]
        try:
            img = self.loader(path)
        except FileNotFoundError:
            #print("Image not found: %s"%(path))
            return
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return  img, target, ids
        #return

    def collate(self, batch):
        #pdb.set_trace()
        batch = list(filter (lambda x:x is not None and x[0] is not None and x[1] is not None and x[2] is not None, batch))
        #return dataloader.default_collate(batch)
        if len(batch) != 0:
            data = torch.stack([item[0] for item in batch])
            target = tuple([item[1] for item in batch])
            ids = tuple([item[2] for item in batch])
            #target = torch.FloatTensor(target)
            return [data, target, ids]
        else:
            return None