import torch
import numpy as np

from src.model.segmentation.unet import UNet
from src.model.segmentation.prob_unet import ProbabilisticUNetWrapper, ProbabilisticModule, OutputModule
from src.dataset import BoeChiuFluidSegDataset
from src.trainer import TrainerForProbUNet

from torchmetrics.classification import BinaryJaccardIndex


def get_pairs(x: torch.Tensor, y: torch.Tensor):
    x_num = x.size(0)
    y_num = y.size(0)
    pairwise_x = torch.broadcast_to(x.unsqueeze(1), size=(x.size(0), y_num, *x.size()[1:]))
    pairwise_y = torch.broadcast_to(y.unsqueeze(0), size=(x_num, *y.size()))
    return pairwise_x, pairwise_y


def energy_distance(manual1, manual2, sampling):
    def distance(x, y):
        iou_func = BinaryJaccardIndex()
        return 1 - iou_func(x, y)

    truth = torch.stack([manual1, manual2])
    sampling = sampling.argmax(dim=2)

    pair_truth1, pair_truth2 = get_pairs(truth, truth)
    pair_sampling1, pair_sampling2 = get_pairs(sampling, sampling)
    pair_cross1, pair_cross2 = get_pairs(truth, sampling)

    score = 2 * distance(pair_cross1.contiguous().view(-1, *pair_cross1.size()[2:]), pair_cross2.contiguous().view(-1, *pair_cross2.size()[2:])) - distance(pair_truth1.contiguous().view(-1, *pair_truth1.size()[2:]), pair_truth2.contiguous().view(-1, *pair_truth2.size()[2:])) - distance(pair_sampling1.contiguous().view(-1, *pair_sampling1.size()[2:]), pair_sampling2.contiguous().view(-1, *pair_sampling2.size()[2:]))
    return score


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

    # get energy score
    manual1 = np.load('/srv/disk00/junhal11/oct_understanding/data/2015_boe_chiu/2015_BOE_Chiu/manual_1.npz')
    manual2 = np.load('/srv/disk00/junhal11/oct_understanding/data/2015_boe_chiu/2015_BOE_Chiu/manual_2.npz')

    score = 0
    num = 9
    for i in range(num):
        sub_score = energy_distance(
            torch.from_numpy(manual1['y_train'][i]),
            torch.from_numpy(manual2['y_train'][i]),
            torch.stack(train_samplings['y_pred'][i]['sampling_pred']).cpu()
        )
        score += torch.nan_to_num(sub_score).item()

    trainer.logger.info(f'Energy score: {score / num: .3f}')
