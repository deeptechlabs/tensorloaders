from tensorloaders import RedisDataset

import redis
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np


def test_redis_dataset():
    # Create random data and push to redis
    r = redis.Redis(host='localhost', port=6379, db=0)

    nb_images = 100
    for idx in range(nb_images):
        # Use long for the fake images
        # as it's easier to store the target with it
        data = np.random.randint(0, 256, (3, 24, 24), dtype=np.long).tobytes()
        target = bytes(np.random.randint(0, 10, (1,)).astype(np.long))
        r.set(idx, data + target)

    # Load samples from redis using multiprocessing
    dataset = RedisDataset(length=100, transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=2,
        shuffle=True
    )
    try:
        for data, target in loader:
            print(data.shape)
            print(target.shape)
    except KeyError:
        assert False

    assert True
