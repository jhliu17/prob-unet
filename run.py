import os
import time
import torch

from src.train import train_unet
from src.args import get_common_parser

if __name__ == '__main__':
    parser = get_common_parser()
    args = parser.parse_args()

    # reproducible
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # post-processing arguments
    current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S",
        time.localtime(time.time())
    )
    experiment_name = f'{args.mode}-{current_time}'
    experiment_path = os.path.join(args.ckpt_dir, experiment_name)
    args.ckpt_dir = experiment_path
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train_unet(args=args)
