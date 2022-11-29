import os 
import torch
import pandas as pd
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    SubsetRandomSampler,
    get_worker_info,
)
from PIL import Image

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        df = pd.read_csv(input_filename, sep=sep)

        self.location = os.path.dirname(input_filename)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.location, str(self.images[idx]))
        images = self.transforms(Image.open(image_path))
        return images

class ANIMAL:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=32,
        num_workers=16,
        classnames=None,
    ):
        file_name = MMY
        file_path = os.path.join(location, file_name)
        self.template = lambda c: f"a photo of a {c}."
        self.train_dataset = CsvDataset(
            input_filename=file_path,
            transforms=transforms,
            img_key="filepath",
            caption_key="title",
        )
        # breakpoint()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )