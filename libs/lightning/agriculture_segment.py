#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: agriculture_segment.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 23:16
import numpy as np
import torch
import torchmetrics.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from libs.archs import resnet
from libs.loss.acw_loss import ACWLoss
from libs.metrics.m import evaluate


class AgricultureSegmentClass(LightningModule):
    def __init__(self, model_name, num_classes):
        """
        所有变量将会自动放置在device上
        :param model_name:
        :param num_classes:
        """
        super(AgricultureSegmentClass, self).__init__()
        self.model = resnet.__dict__[model_name](num_classes=num_classes)
        self.num_classes = num_classes
        self.save_hyperparameters()
        self.lr = 1.5e-4 / np.sqrt(3)
        self.loss = ACWLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=2e-5)
        scheduler = CosineAnnealingLR(opt, 60, 1.18e-6)
        return {'optimizer': opt, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        inputs, gts = batch[0], batch[1]
        outputs, cost = self(inputs)
        main_loss = self.loss(outputs, gts)
        loss = main_loss + cost
        metrix = {"loss": main_loss, "aux_loss": cost}
        self.log_dict(metrix, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # 参考https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # 返回loss，在其背后，会对每个batch的输出堆叠起来，最后求均值
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, gts = batch
        # N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        outputs = self(inputs)
        loss = self.loss(outputs, gts)

        # outputs: torch.Size([10, 7, 512, 512]), 7个类别
        # outputs.data.max(1) -> torch.return_types.max(values=tensor(...), indices=tensor(...))
        #         返回dim=1上最大元素和最大元素对应的索引，values张量表示dim=1维度上的最大元素，indices张量表示对应的索引
        #         outputs.data.max(1)[1] 即返回每张图的所有像素点的位置上值最大的通道索引，这里的通道即语义类别
        # 具体参考 https://blog.csdn.net/weixin_44770969/article/details/124603847

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu()
        preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
        gts = gts.data.squeeze(0).cpu().numpy()
        self.validation_step_outputs.append((preds, gts))
        return loss

    def on_validation_epoch_end(self):
        """
        在on_train_epoch_end方法之前执行
        :return:
        """
        # preds1 = [i[0] for i in self.validation_step_outputs]
        # gts1 = [i[1] for i in self.validation_step_outputs]
        # preds, gts = torch.cat(preds1), torch.cat(gts1)

        acc, acc_cls, miou, fwavacc, f1 = evaluate(self.validation_step_outputs, self.num_classes)  # 准确率，每一类的准确率，miou，加权准确率
        metrics = {"acc": acc, "acc_cls": acc_cls, "miou": miou, "fwavacc": fwavacc, "f1score": f1}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

