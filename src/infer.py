import torch

from src.model.segmentation.unet import UNet
from src.model.segmentation.prob_unet import ProbabilisticUNetWrapper, ProbabilisticModule, OutputModule
from src.dataset import BoeChiuFluidSegDataset
from src.trainer import TrainerForProbUNet


def sample_from_prob_unet(args):
    train_dataset = BoeChiuFluidSegDataset(args.train_dataset_path)
    test_dataset = BoeChiuFluidSegDataset(args.test_dataset_path)

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

    model.load_state_dict(torch.load(args.load_ckpt_path, map_location=torch.device('cpu')))
    trainer = TrainerForProbUNet(args, model)
    trainer.logger.info('Dataset details:')
    trainer.logger.info(f'train: {len(train_dataset)} test: {len(test_dataset)}')

    trainer.logger.info('Start sampling on train dataset')
    train_samplings = trainer.sample(train_dataset, 5)
    trainer.logger.info('Start sampling on test dataset')
    test_samplings = trainer.sample(test_dataset, 5)
    torch.save(train_samplings, f'{args.ckpt_dir}/train_samplings.pth')
    torch.save(test_samplings, f'{args.ckpt_dir}/test_samplings.pth')
