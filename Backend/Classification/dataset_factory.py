from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split
import os

class DatasetFactory:
    def __init__(self, batch_size=64, val_split=0.3, blur=False, flip=False, rotate=False):
        self.batch_size = batch_size
        self.val_split = val_split
        self.use_blur = blur
        self.use_flip = flip
        self.use_rotate = rotate

    def get_dataset(self, dataset_path):
        # Build augmentation list
        augmentations = [transforms.Resize((224, 224))]
        
        if self.use_flip:
            augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
            print("Using Horizontal Flip Augmentation")
        
        if self.use_rotate:
            augmentations.append(transforms.RandomRotation(degrees=15))
            print("Using Rotation Augmentation")
        
        if self.use_blur:
            augmentations.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
            print("Using Gaussian Blur Augmentation")

        augmentations.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        transform = transforms.Compose(augmentations)
        
        # Validation transform without augmentations
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        if dataset_path.lower() == 'cifar10':
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
            classes = trainset.classes
        elif os.path.isdir(dataset_path):
            full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
            # Create a separate dataset for validation with val_transform
            val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
            
            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size
            
            trainset, _ = random_split(
                full_dataset, 
                [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            _, valset = random_split(
                val_dataset, 
                [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            classes = full_dataset.classes
        else:
            raise ValueError(f"Unknown dataset or invalid path: {dataset_path}")

        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)

        num_classes = len(classes)
        return train_loader, val_loader, num_classes, classes

