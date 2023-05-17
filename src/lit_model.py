import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from src.model.segmentation.unet import UNet


class LitUNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = UNet(in_channel=1, out_channel=2)

    def training_step(self, batch, batch_idx):
        tensor_x, tensor_y = batch
        output = self.model(tensor_x)
        # output shape =(batch_size, n_classes, img_cols, img_rows)
        output = output.permute(0, 2, 3, 1).contiguous()  # [b, ]
        # output shape =(batch_size, img_cols, img_rows, n_classes)

        output = output.view(-1, 2)
        label = tensor_y.view(-1)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, label)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
