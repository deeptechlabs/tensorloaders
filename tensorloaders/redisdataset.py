"""
Shows how to store and load data from redis using a PyTorch
Dataset and DataLoader (with multiple workers).
@author: ptrblck
"""

import redis

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np

# Create RedisDataset
class RedisDataset(Dataset):
    def __init__(self,
                 redis_host='localhost',
                 redis_port=6379,
                 redis_db=0,
                 length=0,
                 transform=None):

        self.db = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.length = length
        self.transform = transform

    def __getitem__(self, index):
        data = self.db.get(index)
        data = np.frombuffer(data, dtype=np.long)
        x = data[:-1].reshape(3, 24, 24).astype(np.uint8)
        y = torch.tensor(data[-1]).long()
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.length
