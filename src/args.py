import argparse


def get_common_parser():
    parser = argparse.ArgumentParser()

    # code related
    parser.add_argument('--mode', type=str,
                        choices=['train', 'eval'], default='train',
                        help='code running mode')
    parser.add_argument('--seed', type=int, default=2023,
                        help='code running seed')
    parser.add_argument('--git_info', type=str,
                        help='code running commit hash')

    # input related
    parser.add_argument('--train_dataset_path', type=str, default='',
                        help='dataset path')
    parser.add_argument('--test_dataset_path', type=str, default='',
                        help='dataset path')
    parser.add_argument('--load_ckpt_dir', type=str,
                        help='model loading checkpoint file')
    parser.add_argument('--load_ckpt_path', type=str,
                        help='model loading checkpoint file')

    # output related
    parser.add_argument('--ckpt_dir', type=str, default='output',
                        help='checkpoint path')

    # training related
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size during training stage')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='batch size during evaluation stage')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lambda1', type=float, default=1,
                        help='loss weight for sign loss')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for updating learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--number_worker', type=int, default=0,
                        help='worker number in the training dataloaders')
    parser.add_argument('--eval_number_worker', type=int, default=0,
                        help='worker number in the evaluating dataloaders')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='frequence for logging out')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='frequence for evaluating model during training stage')
    parser.add_argument('--save_ratio', type=float, default=0.5,
                        help='ratio for starting model saving during training stage')
    return parser
