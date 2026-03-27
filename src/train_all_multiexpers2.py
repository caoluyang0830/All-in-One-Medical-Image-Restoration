from typing import List

import os
import pathlib
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import Subset
# from net.AMIFound_2 import NestedAMIFound
from net.AMIFound import AMIFound_Multiexpers

from options2 import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils_all import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

        # self.net = NestedAMIFound(
        #     dim=opt.dim,
        #     num_blocks=opt.num_blocks,
        #     num_dec_blocks=opt.num_dec_blocks,
        #     levels=len(opt.num_blocks),
        #     heads=opt.heads,
        #     num_refinement_blocks=opt.num_refinement_blocks,
        #     topk=opt.topk,
        #     num_experts=opt.num_exp_blocks,
        #     rank=opt.latent_dim,
        #     with_complexity=opt.with_complexity,
        #     depth_type=opt.depth_type,
        #     stage_depth=opt.stage_depth,
        #     rank_type=opt.rank_type,
        #     complexity_scale=opt.complexity_scale, )
        # self.net = AMIFound(
        #     dim=opt.dim,
        #     num_blocks=opt.num_blocks,
        #     num_dec_blocks=opt.num_dec_blocks,
        #     levels=len(opt.num_blocks),
        #     heads=opt.heads,
        #     num_refinement_blocks=opt.num_refinement_blocks,
        #     # 替换原有单层专家参数为双层结构参数
        #     # num_principals=opt.num_principals,  # 新增：校长数量（对应原专家层级）
        #     # top_k_principals=opt.top_k_principals,  # 新增：每个学生选几个校长
        #     # num_teachers_per_principal=opt.num_experts,  # 映射：原num_experts变为每个校长手下的老师数量
        #     top_k_teachers=opt.topk,  # 映射：原topk变为每个校长下选几个老师
        #     teacher_rank=opt.latent_dim,  # 映射：原rank变为老师的中间维度
        #     teacher_depth=opt.stage_depth,  # 映射：原stage_depth变为老师的处理深度
        #     with_complexity=opt.with_complexity,
        #     complexity_scale=opt.complexity_scale,
        # )
        self.net = AMIFound_Multiexpers(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale, )
        if opt.loss_type == "fft":
            self.loss_fn = nn.L1Loss()
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)
        else:
            self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        if self.opt.loss_type == "fft":
            loss = self.loss_fn(restored, clean_patch)
            aux_loss = self.aux_fn(restored, clean_patch)
            loss += aux_loss
        else:
            loss = self.loss_fn(restored, clean_patch)

        loss += self.balance_loss_weight * balance_loss
        self.log("Train_Loss", loss, sync_dist=True)
        self.log("Balance", balance_loss, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=1, max_epochs=self.opt.epochs)
        return [optimizer], [scheduler]


def main(opt):
    print("Options")
    print(opt)
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # --------------------------
    # 关键修正：统一导入风格为lightning.pytorch
    # --------------------------
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, Callback

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt)
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    # 确保ModelCheckpoint来自lightning.pytorch
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=5, save_top_k=-1, save_last=True)

    # 修正：自定义回调类继承自lightning.pytorch的Callback
    class UpdateDatasetEpochCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            if hasattr(trainer.train_dataloader.dataset, 'set_epoch'):
                trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)

    # Create datasets and dataloaders
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
        trainset.set_epoch(0)  # 初始化第一个epoch的样本

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True,
                             shuffle=True, drop_last=True, num_workers=opt.num_workers)

    # Create trainer - 使用统一的lightning.pytorch.Trainer
    trainer = Trainer(max_epochs=opt.epochs,
                      accelerator="gpu",
                      devices=opt.num_gpus,
                      strategy="ddp_find_unused_parameters_true",
                      logger=logger,
                      callbacks=[checkpoint_callback, UpdateDatasetEpochCallback()],
                      accumulate_grad_batches=opt.accum_grad,
                      deterministic=True)

    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = "/caoluyang/code/AMIFound-main/checkpoints/AMIFound_AIO5/last.ckpt"
    else:
        checkpoint_path = None

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        ckpt_path=checkpoint_path
    )


if __name__ == '__main__':
    print("Training Start")
    train_opt = train_options()
    main(train_opt)


