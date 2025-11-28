from base import BaseDataSet, BaseDataLoader
from glob import glob
import numpy as np
import os
from PIL import Image


class CVATBasedDataset(BaseDataSet):
    def __init__(self, palette_file=None, **kwargs):
        # Parse palette file before calling super().__init__
        self.num_classes, self.palette, self.label_map = self._parse_palette_file(palette_file)
        super(CVATBasedDataset, self).__init__(**kwargs)

    def _parse_palette_file(self, palette_file):
      """
      Parse palette file with format:
      # label:color_rgb:parts:actions
      background:0,0,0::
      hole:51,91,102::
      """
      if palette_file is None or not os.path.exists(palette_file):
          raise ValueError(f"Palette file not found: {palette_file}")

      entries = []
      with open(palette_file, 'r') as f:
          for line in f:
              line = line.strip()
              if not line or line.startswith('#'):
                  continue
              parts = line.split(':')
              if len(parts) >= 2:
                  label_name = parts[0].strip()
                  color_str = parts[1].strip()
                  rgb = [int(c.strip()) for c in color_str.split(',')]
                  entries.append((label_name, rgb))

      # set background as the first class
      entries.sort(key=lambda x: 0 if x[0].lower() == "background" else 1)

      labels = [e[0] for e in entries]
      colors = [c for e in entries for c in e[1]]

      num_classes = len(labels)
      palette = colors
      label_map = {label: idx for idx, label in enumerate(labels)}

      return num_classes, palette, label_map

    def _set_files(self):
        image_dir = os.path.join(self.root, 'images', self.split)
        mask_dir = os.path.join(self.root, 'masks', self.split)
        
        image_paths = sorted(glob(os.path.join(image_dir, '*.*')))
        mask_paths = sorted(glob(os.path.join(mask_dir, '*.*')))
        
        assert len(image_paths) == len(mask_paths), "Number of images and masks do not match."
        self.files = list(zip(image_paths, mask_paths))

    def _load_data(self, index):
        image_path, mask_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask = np.asarray(Image.open(mask_path).convert('RGB'), dtype=np.int32)
        
        # Convert RGB mask to class indices
        # Initialize with -1 (ignore_index) for unmatched pixels
        label = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.int32)
        
        # Reshape palette to (num_classes, 3) for easier comparison
        palette_rgb = np.array(self.palette).reshape(-1, 3)
        
        for class_idx in range(self.num_classes):
            color = palette_rgb[class_idx]
            # Find all pixels matching this color
            mask_class = np.all(mask == color, axis=2)
            label[mask_class] = class_idx
        
        return image, label, image_id


class CVATBased(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, palette_file, crop_size=None, base_size=None, 
                 scale=True, num_workers=1, val=False, shuffle=False, flip=False, rotate=False, 
                 blur=False, augment=False, val_split=None, return_id=False, mean=None, std=None):

        # Use provided mean/std or defaults
        self.MEAN = mean if mean is not None else [0.485, 0.456, 0.406]
        self.STD = std if std is not None else [0.229, 0.224, 0.225]

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
            'val': val,
            'palette_file': palette_file
        }

        self.dataset = CVATBasedDataset(**kwargs)
        super(CVATBased, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
