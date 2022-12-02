import torch.utils.data
from dataset.base_dataset import BaseDataset
from dataset.unaligned_dataset import UnalignedDataset
import os
from types import SimpleNamespace


def create_dataset(opt):
    opt = SimpleNamespace(**opt)
    data_loader_train = CustomDatasetDataLoader(opt, phase="train")
    dataset_train = data_loader_train.load_data()
    data_loader_val = CustomDatasetDataLoader(opt, phase="test")
    dataset_val = data_loader_val.load_data()
    return dataset_train, dataset_val


class CustomDatasetDataLoader:
    def __init__(self, opt, phase: str):
        self.opt = opt
        self.dataset = UnalignedDataset(opt, phase)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
