import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import pkload


class OASISBrainDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        z = pkload(path)
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class IXIBrainDataset(Dataset):

    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):

    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class CTMRIABDDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x = np.load(path)
        x_seg = np.load(path.replace("data", "mask"))

        y_data_file = os.path.dirname(path.replace("CT", "MRI"))
        y_data_path = y_data_file + '/' + random.choice(
            os.listdir(y_data_file))
        y = np.load(y_data_path)
        y_seg = np.load(y_data_path.replace("data", "mask"))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class CTMRIABDInferDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = np.load(path)
        x_seg = np.load(path.replace("data", "mask"))

        y = np.load(path.replace("CT", "MRI"))
        y_seg = np.load(path.replace("CT", "MRI").replace("data", "mask"))
        #x = np.pad(x, ((0, 0), (0, 0), (0, 2)), 'constant')
        #y = np.pad(y, ((0, 0), (0, 0), (0, 2)), 'constant')
        #x_seg = np.pad(x_seg, ((0, 0), (0, 0), (0, 2)), 'constant')
        #y_seg = np.pad(y_seg, ((0, 0), (0, 0), (0, 2)), 'constant')
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class CMFDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = np.load(path)
        y = np.load(path.replace("CT", "MRI"))
        #y_data_file = os.path.dirname(path.replace("CT","MRI"))
        #y_data_path = y_data_file + '/' + random.choice(os.listdir(y_data_file))
        #y = np.load(y_data_path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x[None, ...], y[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class CMFInferDataset(Dataset):

    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x = np.load(path)
        y = np.load(path.replace("CT", "MRI"))
        x_seg = np.load(path.replace("data", "mask"))
        y_seg = np.load(path.replace("data", "mask").replace("CT", "MRI"))

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(
            x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(
            y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
