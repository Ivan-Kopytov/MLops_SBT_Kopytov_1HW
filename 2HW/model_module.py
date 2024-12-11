import pytorch_lightning as pl

import torch

from torch import nn

import torch.nn.functional as F



class SimpleRegressionModel(pl.LightningModule):

    def __init__(self, lr=1e-3):

        super().__init__()

        # Простейшая линейная модель: y_pred = w*x + b

        self.w = nn.Parameter(torch.randn(1))

        self.b = nn.Parameter(torch.randn(1))

        self.lr = lr



    def forward(self, x):

        return x * self.w + self.b



    def training_step(self, batch, batch_idx):

        x, y, _ = batch  # batch = (X[idx], Y[idx], [b0,b1])

        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y)

        self.log('train_loss', loss)

        return loss



    def validation_step(self, batch, batch_idx):

        x, y, _ = batch

        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True)

        return loss



    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)


