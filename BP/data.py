import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Sampler, SequentialSampler

import jax.numpy as jnp
import jax

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.expand_dims(np.array(pic, dtype=jnp.float32),axis=-1)

# DataLoader返回numpy array，而不是torch Tensor
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class JAXRandomSampler(Sampler):
    def __init__(self, data_source, rng_key):
        self.data_source = data_source
        self.rng_key = rng_key
        
    def __len__(self):
        return len(self.data_source)
    
    def __iter__(self):
        self.rng_key, current_rng = jax.random.split(self.rng_key)
        return iter(jax.random.permutation(current_rng, jnp.arange(len(self))).tolist())


class NumpyLoader(DataLoader):
    def __init__(self, dataset, rng_key=None, batch_size=1,
                 shuffle=False, **kwargs):
        if shuffle:
            sampler = JAXRandomSampler(dataset, rng_key)
        else:
            sampler = SequentialSampler(dataset)
        
        super().__init__(dataset, batch_size, sampler=sampler, **kwargs)


