import torch

from torch.utils.data import random_split

from src.model.segmentation.unet import UNet
from src.dataset import BoeChiuFluidSegDataset
from src.trainer import TrainerForUNet


def train_unet(args):
    train_dataset = BoeChiuFluidSegDataset(args.train_dataset_path)
    test_dataset = BoeChiuFluidSegDataset(args.test_dataset_path)
    train_size = int(len(train_dataset) * 0.7)
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed)
    )

    # init model and trainer
    model = UNet(1, 2)
    trainer = TrainerForUNet(args, model)
    trainer.logger.info('Dataset details:')
    trainer.logger.info(f'train: {len(train_dataset)} valid: {len(valid_dataset)} test: {len(test_dataset)}')

    # start training
    trainer.train(train_dataset, valid_dataset)

    # report performance on testing dataset
    trainer.logger.info("Evaluate test dataset...")
    test_output, _ = trainer.eval(test_dataset, num_workers=args.eval_number_worker)
    for k, v in test_output.items():
        trainer.logger.info(f"{k}: {v}")
