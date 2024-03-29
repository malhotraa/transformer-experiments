import logging
import math
import pickle
import os
from typing import NamedTuple, List, Optional

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class DataSample(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor

class DataBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor

class CharDataset(Dataset):
    DATA_PATH = 'data/input.txt'

    def __init__(self, block_size: int, limit_len: Optional[int] = None) -> None:
        data = open(self.DATA_PATH, 'r').read()
        if limit_len:
            data = data[:block_size + limit_len]
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def encode(self, seq: str) -> torch.Tensor:
        dix = [self.stoi[s] for s in seq]
        x = torch.tensor(dix, dtype=torch.long)
        return x
    
    def decode(self, seq: torch.Tensor) -> str:
        out = []
        for b in seq.tolist():
            out.append(''.join([self.itos[x] for x in b]))
        return out

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> DataSample:
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = self.encode(chunk)
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward
        passes.

        block = hello, ml is great, I am writing somthing


        batch = [(h, e), (he, l), (hel, l), (hell, o), ...]
        """
        x, y = dix[:-1], dix[1:]
        return DataSample(x, y)

    @staticmethod
    def transform(sample: DataSample) -> DataSample:
        pass

    @staticmethod
    def collate_fn(batch_list: List[DataSample]) -> DataBatch:
        x_batch = []
        y_batch = []
        for sample in batch_list:
            x_batch.append(sample.x)
            y_batch.append(sample.y)
        return DataBatch(x=torch.stack(x_batch, dim=0), y=torch.stack(y_batch, dim=0))


class ImageDataset(Dataset):
    def __init__(self):
        root = '.data'
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
        test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
        print(len(train_data), len(test_data))
        CLUSTERS_PATH = '.data/clusters.pkl'
        if not os.path.exists(CLUSTERS_PATH):
            # get random 5 pixels per image and stack them all up as rgb values to get half a million random pixels
            pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:5], :]
            px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()
            print(px.size())

            ncluster = 512
            with torch.no_grad():
                C = ImageDataset.kmeans(px, ncluster, niter=8)
            with open(CLUSTERS_PATH, 'wb') as f:
                pickle.dump(C, f)
        else:
            with open(CLUSTERS_PATH, 'rb') as f:
                C = pickle.load(f)

        # encode the training examples with our codebook to visualize how much we've lost in the discretization
        n_samples = 16
        ncol = 8
        nrow = n_samples // ncol + 1
        plt.figure(figsize=(20, 10))
        for i in range(n_samples):
            # encode and decode random data
            x, y = train_data[np.random.randint(0, len(train_data))]
            xpt = torch.from_numpy(np.array(x)).float().view(32*32, 3)
            ix = ((xpt[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1) # cluster assignments for each pixel
            
            # these images should look normal ideally
            plt.subplot(nrow, ncol, i+1)
            plt.plot(C[ix].view(32, 32, 3).numpy().astype(np.uint8))
        
        plt.save_fig('temp.png')

    @staticmethod
    def kmeans(x, ncluster, niter=10):
        N, D = x.size()
        perm = torch.randperm(N)
        clusters = x[perm[:ncluster]] # init clusters at random; might be duplicates in clusters
        for i in range(niter):
            # assign all pixels to the closest codebook element
            diff = (x[:, None, :] - clusters[None, :, :])
            l2 = diff**2
            a = l2.sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            clusters = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(clusters), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
            clusters[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
        return clusters

def build_synthetic_dataset(block_size, num_samples):
    class SyntheticDataset(Dataset):
        def __init__(self, block_size, num_samples, vocab_size=100):
            self.block_size = block_size
            self.vocab_size = vocab_size
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            data = torch.randint(1, self.vocab_size, (block_size,), requires_grad=False)
            data[0] = 1
            return data, data

        def batch(self, batch_size):
            x_batch = []
            y_batch = []
            for idx in range(batch_size):
                x, y = self.__getitem__(idx)
                x_batch.append(x)
                y_batch.append(y)
            return torch.stack(x_batch, dim=0), torch.stack(y_batch, dim=0)

    return SyntheticDataset(block_size, num_samples)