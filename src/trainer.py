import logging
import os
import torch

from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from torchmetrics.classification import BinaryJaccardIndex
from src.utils import plot_segmentation_examples


class Trainer:
    def __init__(self, args, model):
        # set parsed arguments
        self.args = args

        # init logger and tensorboard
        self._init_logger()
        self._set_writer()

        # init ingredients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # init model
        self.model = model
        self.model = self.model.to(self.device)

        # init optimizer and learning rate scheduler
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr, momentum=0.99)

        # log status
        self.logger.info("Experiment setting:")
        for k, v in sorted(vars(self.args).items()):
            self.logger.info(f"{k}: {v}")

    def resume(self, resume_ckpt_path: str):
        # resume checkpoint
        self.logger.info(f"Resume model checkpoint from {resume_ckpt_path}...")
        self.model.load_state_dict(torch.load(resume_ckpt_path))

    def train(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.train_step)

    def _set_writer(self):
        self.logger.info("Create writer at '{}'".format(self.args.ckpt_dir))
        self.writer = SummaryWriter(self.args.ckpt_dir)

    def _init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.args.ckpt_dir, f"{self.args.mode}.log"),
            level=logging.INFO,
            datefmt="%Y/%m/%d %H:%M:%S",
            format="%(asctime)s: %(name)s [%(levelname)s] %(message)s",
        )
        formatter = logging.Formatter(
            "%(asctime)s: %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)


class TrainerForUNet(Trainer):
    def train_loop(self, train_dataset, eval_dataset, step_func):
        """Training loop function for model training and finetuning.

        :param train_dataset: training dataset
        :param eval_dataset: evaluation dataset
        :param step_func: a callable function doing forward and optimize step and return loss log
        """
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.number_worker,
        )

        global_step = 0
        for epoch in range(0, self.args.epoch):
            # train steps
            for step, (feat, label) in enumerate(train_dataloader):
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)

                # run step
                input_feats = {"feat": feat, "label": label}
                loss_log = step_func(input_feats)

                # print loss
                if step % self.args.log_freq == 0:
                    loss_str = " ".join([f"{k}: {v}" for k, v in loss_log.items()])
                    self.logger.info(f"Epoch: {epoch} Step: {step} | Loss: {loss_str}")
                    for k, v in loss_log.items():
                        self.writer.add_scalar(f"train/{k}", v, global_step)

                    # log current learning rate
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/lr", current_lr, global_step)

                # increase step
                global_step += 1

            if (epoch + 1) % self.args.eval_freq == 0:
                self.logger.info(f"Evaluate eval dataset at epoch {epoch}...")
                eval_output, seg_output = self.eval(eval_dataset)
                for k, v in eval_output.items():
                    self.logger.info(f"{k}: {v}")
                    self.writer.add_scalar(f"train/eval_{k}", v, epoch)

                # plot training segmentation results
                fig, _ = plot_segmentation_examples(
                    seg_output["x"].numpy(),
                    seg_output["y_pred"].numpy(),
                    seg_output["y_true"].numpy(),
                )
                self.writer.add_figure("train/eval_seg", figure=fig)

                torch.save(
                    self.model.state_dict(), f"{self.args.ckpt_dir}/model_{epoch}.pth"
                )

        # save the final model after training
        torch.save(self.model.state_dict(), f"{self.args.ckpt_dir}/model_final.pth")

    def train_step(self, input_feats):
        self.model.train()
        feat, label = input_feats["feat"], input_feats["label"]

        # clean gradient and forward
        self.optimizer.zero_grad()

        output = self.model(feat)
        # output shape =(batch_size, n_classes, img_cols, img_rows)
        output = output.permute(0, 2, 3, 1).contiguous()  # [b, ]
        # output shape =(batch_size, img_cols, img_rows, n_classes)

        output = output.view(-1, 2)
        label = label.view(-1)
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(output, label)

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {"loss": loss.item()}
        return log

    @torch.inference_mode()
    def eval(self, dataset, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        feat_list = []
        y_pred_list = []
        y_true_list = []
        for feat, label in tqdm(dataloader):
            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            feat_list.append(feat.cpu())
            y_pred_list.append(pred.cpu())
            y_true_list.append(label)

        x = torch.cat(feat_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.eval_on_prediction(y_pred, y_true)

        output = {}
        output["x"] = x
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        return score, output

    @torch.inference_mode()
    def eval_on_prediction(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # iou metric
        iou_fn = BinaryJaccardIndex()
        segment = y_pred.argmax(dim=1, keepdim=True)
        iou_score = iou_fn(segment, y_true)
        num1 = segment.sum().item()
        num2 = y_true.sum().item()

        score = {}
        score["iou_score"] = iou_score
        score["pred_area"] = num1
        score["label_area"] = num2
        return score


class TrainerForProbUNet(TrainerForUNet):
    def train_step(self, input_feats):
        self.model.train()
        feat, label = input_feats["feat"], input_feats["label"]

        # clean gradient and forward
        self.optimizer.zero_grad()

        outputs = self.model(feat, label)
        output = outputs['pred']

        # output shape =(batch_size, n_classes, img_cols, img_rows)
        output = output.permute(0, 2, 3, 1).contiguous()  # [b, ]
        # output shape =(batch_size, img_cols, img_rows, n_classes)

        output = output.view(-1, 2)
        label = label.view(-1)
        seg_loss_fn = NLLLoss()
        seg_loss = seg_loss_fn(output, label)
        kl_loss = torch.distributions.kl_divergence(outputs['posterior_dist'], outputs['prior_dist']).mean()
        loss = seg_loss + (kl_loss * 1e-2 if kl_loss < 1e2 else kl_loss * 1e-3)

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {"loss": loss.item(), 'seg_loss': seg_loss.item(), 'kl_loss': kl_loss.item()}
        return log

    @torch.inference_mode()
    def eval(self, dataset, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        feat_list = []
        y_pred_list = []
        y_true_list = []
        elbo = 0
        for feat, label in tqdm(dataloader):
            feat: torch.Tensor = feat.to(self.device)
            label: torch.Tensor = label.to(self.device)

            outputs = self.model(feat, label)
            elbo += self.model.elbo(feat, label)

            pred = outputs['pred']
            feat_list.append(feat.cpu())
            y_pred_list.append(pred.cpu())
            y_true_list.append(label.cpu())

        x = torch.cat(feat_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.eval_on_prediction(y_pred, y_true)
        score.update({'elbo': elbo.item()})

        output = {}
        output["x"] = x
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        return score, output

    @torch.inference_mode()
    def sample(self, dataset, N: int = 1, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )

        feat_list = []
        y_pred_list = []
        y_true_list = []
        for i, (feat, label) in enumerate(tqdm(dataloader)):
            feat: torch.Tensor = feat.to(self.device)
            label: torch.Tensor = label.to(self.device)

            mean_outputs = self.model(feat, label)
            y_pred_list.append({'mean_pred': mean_outputs['pred'].cpu(), 'sampling_pred': []})
            feat_list.append(feat.cpu())
            y_true_list.append(label.cpu())

            sampling_outputs = self.model.sampling_reconstruct(feat, N)
            y_pred_list[-1]['sampling_pred'].extend([i.cpu() for i in sampling_outputs])

        outputs = {
            'x': feat_list,
            'y_pred': y_pred_list,
            'y_true': y_true_list
        }
        return outputs

    @torch.inference_mode()
    def eval_on_prediction(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # iou metric
        iou_fn = BinaryJaccardIndex()
        segment = y_pred.argmax(dim=1, keepdim=True)
        iou_score = iou_fn(segment, y_true)
        num1 = segment.sum().item()
        num2 = y_true.sum().item()

        score = {}
        score["iou_score"] = iou_score
        score["pred_area"] = num1
        score["label_area"] = num2
        return score
