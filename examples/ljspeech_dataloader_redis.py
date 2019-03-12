from tensorloaders import RedisDataset

import redis
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np


def load_ljspeech():
    # Create random data and push to redis
    r = redis.Redis(host='localhost', port=6379, db=1)
    # Load samples from redis using multiprocessing
    dataset = RedisDataset(length=100, transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True
    )
    for data in loader:
        print(data)

if __name__ == "__main__":
    load_ljspeech()
