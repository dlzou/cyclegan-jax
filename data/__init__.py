import torch.utils.data
from data.base_dataset import BaseDataset
from data.unaligned_dataset import UnalignedDataset
import os 
from types import SimpleNamespace

def create_dataset(opt={}):
    opt = {
        'dataset_mode':'unaligned', 
        'max_dataset_size':float("inf"), 
        'preprocess':'resize_and_crop',
        'no_flip':True, 
        'display_winsize': 256,
        'num_threads': 4,
        'batch_size': 16,
        'load_size': 286,
        'crop_size': 256,
        'dataroot': "./horse2zebra", 
        "phase":"train", 
        "direction": "AtoB", 
        "input_nc": 3, 
        "output_nc": 3, 
        "serial_batches": True,
    }
    opt = SimpleNamespace(**opt)
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data



