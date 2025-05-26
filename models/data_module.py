from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms

class ImageDataModule(LightningDataModule):
    def __init__(self, data_dir="data/animals10", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def setup(self, stage=None):
        self.train_dataset = ImageFolder(f"{self.data_dir}/train", transform=self.train_transforms)
        self.val_dataset = ImageFolder(f"{self.data_dir}/val", transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
