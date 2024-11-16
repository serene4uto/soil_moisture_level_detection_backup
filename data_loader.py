import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataLoaderWrapper:
    def __init__(self, train_dir, val_dir, batch_size=32, img_size=(224, 224)):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),  # Data augmentation for training
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(self.img_size,   # Randomly crops and resizes to the original image size
                                 scale=(0.8, 1.2),
                                 ratio=(0.75, 1.33)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader

# # test code here
# root_path = r"D:\daumn\2.code\kaleem_code\dataset"
# data_loader = DataLoaderWrapper(train_dir=root_path+ '/train', val_dir=root_path+ '/test', batch_size=32)
# train_loader, val_loader = data_loader.get_loaders()
