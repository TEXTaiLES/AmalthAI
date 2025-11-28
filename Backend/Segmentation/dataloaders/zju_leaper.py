from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ZJULeaperDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.ZJULeaper_palette
        super(ZJULeaperDataset, self).__init__(**kwargs)

    def _set_files(self):
        
        image_dir = os.path.join(self.root, 'images', self.split)
        mask_dir = os.path.join(self.root, 'masks', self.split)
        
        image_paths = sorted(glob(os.path.join(image_dir, '*.*')))
        mask_paths = sorted(glob(os.path.join(mask_dir, '*.*')))
        
        assert len(image_paths) == len(mask_paths), "Αριθμός εικόνων και μασκών δεν ταιριάζει."
        self.files = list(zip(image_paths, mask_paths))

    def _load_data(self, index):
        image_path, mask_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(mask_path).convert('L'), dtype=np.int32)
        label = (label == 255).astype(np.int32)
        return image, label, image_id


class ZJULeaper(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = ZJULeaperDataset(**kwargs)
        super(ZJULeaper, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)