# from .animal import ANIMAL
# import jax.numpy as jnp

# # Note: this code is from https://github.com/google/flax/blob/acb4ce0564b76f735799d19f516c1106fee9d3cb/examples/mnist/train.py#L98
# # Decide if we want to follow this convention or change 
# def get_datasets():
#   """Load MNIST train and test datasets into memory."""
#   train_ds = None 
#   test_ds = None 
#   train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#   test_ds['image'] = jnp.float32(test_ds['image']) / 255.
#   return train_ds, test_ds


import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import os 


def find_dataset_using_name(dataset_name):

    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataset(opt):
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset



class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt

        location= os.path.expanduser("~/data")
        file_name = "horse2zebra"
        file_path = os.path.join(location, file_name)
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data



