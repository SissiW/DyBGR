import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import collections


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=np.int)
    return labels_to_indices


def safe_random_choice(input_data, size):
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace).tolist()


class UniqueClassSempler(Sampler):
    def __init__(self, labels, m_per_class, rank=0, world_size=1, seed=0): 
        # print('1', labels) # [0-99]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels_to_indices = get_labels_to_indices(labels)
        # {0: array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #             17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        #             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        #             51, 52, 53, 54, 55, 56, 57, 58, 59]), 1: array([ 60,  61, ..., 5863])})
        self.labels = sorted(list(self.labels_to_indices.keys()))  # sort [0-99]
        self.m_per_class = m_per_class  # 9
        self.rank = rank
        self.world_size = world_size  # 1
        self.seed = seed  # 0
        self.epoch = 0

    def __len__(self):
        return (len(self.labels) // self.world_size) * self.m_per_class

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)
        idx = torch.randperm(len(self.labels), generator=g).tolist()  # len=100
        # idx1 = torch.randperm(len(self.labels), generator=g).tolist()
        # idx = idx0 + idx1

        size = len(self.labels) // self.world_size
        idx = idx[size * self.rank : size * (self.rank + 1)]
        for i in idx:  # i=0-99
            t = self.labels_to_indices[self.labels[i]]
            idx_list += safe_random_choice(t, self.m_per_class)

        # print('*************2', len(idx_list), idx_list)
        return iter(idx_list)  # len=900

    def set_epoch(self, epoch):
        self.epoch = epoch
