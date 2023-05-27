import torch

from torch.utils.data import random_split

from src.model.segmentation.unet import UNet
from src.model.segmentation.prob_unet import ProbabilisticUNetWrapper, ProbabilisticModule, OutputModule
from src.dataset import BoeChiuFluidSegDataset
from src.trainer import TrainerForUNet, TrainerForProbUNet


def split_dataset(dataset, train_ratio, seed):
    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, valid_dataset


def train_unet(args):
    train_dataset = BoeChiuFluidSegDataset(args.train_dataset_path)
    test_dataset = BoeChiuFluidSegDataset(args.test_dataset_path)
    train_dataset, valid_dataset = split_dataset(train_dataset, 0.7, args.seed)

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


def train_prob_unet(args):
    train_dataset = BoeChiuFluidSegDataset(args.train_dataset_path)
    test_dataset = BoeChiuFluidSegDataset(args.test_dataset_path)
    train_dataset, valid_dataset = split_dataset(train_dataset, 0.7, args.seed)

    # init model and trainer
    unet = UNet(1, 8)
    prior_net = ProbabilisticModule(1, args.latent_size)
    posterior_net = ProbabilisticModule(2, args.latent_size)
    output_net = OutputModule(8 + args.latent_size // 2, 32, 2)
    model = ProbabilisticUNetWrapper(
        unet=unet,
        prior_net=prior_net,
        posterior_net=posterior_net,
        output_net=output_net,
        latent_distribution_cls=torch.distributions.Normal
    )
    trainer = TrainerForProbUNet(args, model)
    trainer.logger.info('Dataset details:')
    trainer.logger.info(f'train: {len(train_dataset)} valid: {len(valid_dataset)} test: {len(test_dataset)}')

    # start training
    trainer.train(train_dataset, valid_dataset)

    # report performance on testing dataset
    trainer.logger.info("Evaluate test dataset...")
    test_output, _ = trainer.eval(test_dataset, num_workers=args.eval_number_worker)
    for k, v in test_output.items():
        trainer.logger.info(f"{k}: {v}")
