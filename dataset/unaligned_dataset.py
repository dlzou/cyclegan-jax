import os
from dataset.base_dataset import BaseDataset, get_transform

# from data.image_folder import make_dataset
from PIL import Image
import random

############################ helpers

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


############################


class UnalignedDataset(BaseDataset):
    def __init__(self, opt, phase: str):

        BaseDataset.__init__(self, opt, phase)
        self.dir_A = os.path.join(
            opt.dataroot, self.phase + "A"
        )  # phase in 'train, val, test, etc'
        self.dir_B = os.path.join(opt.dataroot, self.phase + "B")

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input images
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range

        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # breakpoint()

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)


# class UnalignedDatasetVal(BaseDataset):
#     def __init__(self, opt):

#         BaseDataset.__init__(self, opt)
#         self.dir_A = os.path.join(
#             opt.dataroot, opt.phase + "A"
#         )  # phase in 'train, val, test, etc'
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

#         self.A_paths = sorted(
#             make_dataset(self.dir_A, opt.max_dataset_size)
#         )  # load images from '/path/to/data/trainA'
#         self.B_paths = sorted(
#             make_dataset(self.dir_B, opt.max_dataset_size)
#         )  # load images from '/path/to/data/trainB'
#         self.A_size = len(self.A_paths)  # get the size of dataset A
#         self.B_size = len(self.B_paths)  # get the size of dataset B

#         self.A_size_train = int(self.A_size * opt.train_set_ratio)
#         self.A_size_val = self.A_size - self.A_size_train
#         self.B_size_train = int(self.B_size * opt.train_set_ratio)
#         self.B_size_val = self.B_size - self.B_size_train

#         self.A_paths_train,  self.A_paths_val = self.A_paths[:self.A_size_train], self.A_paths[self.A_size_train:]
#         self.B_paths_train,  self.B_paths_val = self.B_paths[:self.B_size_train], self.B_paths[self.B_size_train:]

#         btoA = self.opt.direction == "BtoA"
#         input_nc = (
#             self.opt.output_nc if btoA else self.opt.input_nc
#         )  # get the number of channels of input images
#         output_nc = (
#             self.opt.input_nc if btoA else self.opt.output_nc
#         )  # get the number of channels of output image
#         self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
#         self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

#     def __getitem__(self, index):
#         """Return a data point and its metadata information.

#         Parameters:
#             index (int)      -- a random integer for data indexing

#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor)       -- an image in the input domain
#             B (tensor)       -- its corresponding image in the target domain
#             A_paths (str)    -- image paths
#             B_paths (str)    -- image paths
#         """
#         A_path_train = self.A_paths_train[
#             index % self.A_size_train
#         ]  # make sure index is within then range
#         A_path_val = self.A_paths_val[
#             index % self.A_size_val
#         ]  # make sure index is within then range

#         if self.opt.serial_batches:  # make sure index is within then range
#             index_B_train = index % self.B_size_train
#             index_B_val = index % self.B_size_val
#         else:  # randomize the index for domain B to avoid fixed pairs.
#             index_B_train = random.randint(0, self.B_size_train - 1)
#             index_B_val = random.randint(0, self.B_size_val - 1)

#         B_path_train = self.B_paths_train[index_B_train]
#         B_path_val = self.B_paths_val[index_B_val]

#         A_img_train = Image.open(A_path_train).convert("RGB")
#         A_img_val = Image.open(A_path_val).convert("RGB")
#         B_img_train = Image.open(B_path_train).convert("RGB")
#         B_img_val = Image.open(B_path_val).convert("RGB")
#         # apply image transformation
#         A_train = self.transform_A(A_img_train)
#         A_val = self.transform_A(A_img_val)
#         B_train = self.transform_B(B_img_train)
#         B_val = self.transform_B(B_img_val)
#         # breakpoint()

#         return {"A": A_val, "B": B_val, "A_paths": A_path_val, "B_paths": B_path_val}

#     def __len__(self):
#         return max(self.A_size_val, self.B_size_val)
