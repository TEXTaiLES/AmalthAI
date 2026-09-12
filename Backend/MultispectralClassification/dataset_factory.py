import os
import random

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


TIFF_EXTENSIONS = {".tif", ".tiff"}


def load_multispectral_tensor(path, in_channels):
    array = tifffile.imread(path)
    if array.ndim == 2 and in_channels == 1:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[0] == in_channels:
        pass
    elif array.ndim == 3 and array.shape[-1] == in_channels:
        array = array.transpose(2, 0, 1)
    else:
        raise ValueError(
            f"{path} has shape {array.shape}; expected {in_channels} bands in CHW or HWC layout"
        )

    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        tensor = torch.from_numpy(array.astype(np.float32)).div_(maximum)
    else:
        tensor = torch.from_numpy(array.astype(np.float32))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return tensor


def discover_samples(root):
    classes = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    )
    class_to_idx = {name: index for index, name in enumerate(classes)}
    samples = []
    for class_name in classes:
        class_dir = os.path.join(root, class_name)
        for filename in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, filename)
            if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in TIFF_EXTENSIONS:
                samples.append((path, class_to_idx[class_name]))
    return samples, classes


class MultispectralDataset(Dataset):
    def __init__(self, samples, in_channels, mean, std, augment=False,
                 blur=False, flip=False, rotate=False, scale=False):
        self.samples = samples
        self.in_channels = in_channels
        self.mean = mean
        self.std = std
        self.augment = augment
        self.blur = blur
        self.flip = flip
        self.rotate = rotate
        self.scale = scale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = load_multispectral_tensor(path, self.in_channels)
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.augment:
            if self.flip and random.random() < 0.5:
                image = TF.hflip(image)
            if self.rotate:
                image = TF.rotate(image, random.uniform(-15, 15), interpolation=InterpolationMode.BILINEAR)
            if self.scale:
                scale = random.uniform(0.8, 1.2)
                image = TF.affine(image, angle=0, translate=[0, 0], scale=scale, shear=[0.0, 0.0],
                                  interpolation=InterpolationMode.BILINEAR)
            if self.blur:
                image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 2.0])
        image = TF.normalize(image, self.mean, self.std)
        return image, label


def compute_channel_stats(samples, in_channels):
    channel_sum = torch.zeros(in_channels, dtype=torch.float64)
    channel_sq_sum = torch.zeros(in_channels, dtype=torch.float64)
    pixel_count = 0
    for path, _ in samples:
        image = load_multispectral_tensor(path, in_channels).double()
        channel_sum += image.sum(dim=(1, 2))
        channel_sq_sum += image.square().sum(dim=(1, 2))
        pixel_count += image.shape[1] * image.shape[2]
    if pixel_count == 0:
        raise ValueError("The training split contains no TIFF samples")
    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    return mean.float().tolist(), std.float().tolist()


class DatasetFactory:
    def __init__(self, in_channels, batch_size=64, val_split=0.3, blur=False,
                 flip=False, rotate=False, scale=False, dataset_already_split=False):
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.val_split = val_split
        self.blur = blur
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        self.dataset_already_split = dataset_already_split

    def get_dataset(self, dataset_path):
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Invalid dataset path: {dataset_path}")

        if self.dataset_already_split:
            train_samples, classes = discover_samples(os.path.join(dataset_path, "train"))
            val_samples, val_classes = discover_samples(os.path.join(dataset_path, "val"))
            if classes != val_classes:
                raise ValueError("Train and validation classes do not match")
        else:
            samples, classes = discover_samples(dataset_path)
            generator = torch.Generator().manual_seed(42)
            order = torch.randperm(len(samples), generator=generator).tolist()
            val_size = int(len(samples) * self.val_split)
            val_indices = set(order[:val_size])
            train_samples = [sample for i, sample in enumerate(samples) if i not in val_indices]
            val_samples = [sample for i, sample in enumerate(samples) if i in val_indices]

        if not train_samples or not val_samples:
            raise ValueError("Both training and validation splits must contain TIFF samples")
        mean, std = compute_channel_stats(train_samples, self.in_channels)
        trainset = MultispectralDataset(
            train_samples, self.in_channels, mean, std, augment=True,
            blur=self.blur, flip=self.flip, rotate=self.rotate, scale=self.scale,
        )
        valset = MultispectralDataset(val_samples, self.in_channels, mean, std)
        return (
            DataLoader(trainset, batch_size=self.batch_size, shuffle=True),
            DataLoader(valset, batch_size=self.batch_size, shuffle=False),
            len(classes), classes, mean, std,
        )
