import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import RegressionDataset

class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, train_X, train_Y, val_X, val_Y, batch_size=16):
        super().__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Разбиваем данные на train/val датасеты
        if stage == 'fit' or stage is None:
            self.train_dataset = RegressionDataset(self.train_X, self.train_Y)
            self.val_dataset = RegressionDataset(self.val_X, self.val_Y)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
