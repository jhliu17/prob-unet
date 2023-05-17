import lightning.pytorch as pl

from torch.utils.data import DataLoader
from src.lit_model import LitUNet
from src.dataset import BoeChiuFluidSegDataset


def train_unet(args):
    train_dataset = BoeChiuFluidSegDataset(args.train_dataset)
    train_loader = DataLoader(train_dataset)

    model = LitUNet(args)
    trainer = pl.Trainer(max_epochs=500)
    trainer.fit(model=model, train_dataloaders=train_loader)
