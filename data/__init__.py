import torch.utils.data
from data.base_dataset import BaseDataset
from data.unaligned_dataset import UnalignedDataset_train, UnalignedDataset_val
import os
from types import SimpleNamespace


def create_dataset(opt={}):
    opt = SimpleNamespace(**opt)
    data_loader_train = CustomDatasetDataLoaderTrain(opt)
    dataset_train = data_loader_train.load_data()
    data_loader_val = CustomDatasetDataLoaderVal(opt)
    dataset_val = data_loader_val.load_data()
    return dataset_train, dataset_val


class CustomDatasetDataLoaderTrain:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset_train(opt)
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

class CustomDatasetDataLoaderVal:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset_val(opt)
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

