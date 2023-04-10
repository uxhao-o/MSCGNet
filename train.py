#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: train.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 13:42
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from libs.data.dataset import prepare_gt
from libs.lightning import AgricultureSegmentClass, AgricultureDataClass  # noqa: F401


if __name__ == '__main__':
    # #
    # DATASET_NAME = 'Agriculture'
    # DATASET_ROOT = "D:/Agriculture-Vision"
    # MODEL_NAME = "MSCG_Rx50"
    # NUM_CLASSES = 7
    # CKPT_PATH = './output'
    # MAX_EPOCHS = 10
    # PRECISION = 16
    #
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=os.path.join(CKPT_PATH, 'models'),
    #                                       filename='{epoch:02d}-{loss:.3f}-{aux_loss:.3f}-{val_loss:.3f}-{acc:.3f}-{miou:.3f}-{f1score:.3f}')
    # trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator='gpu', devices=1, log_every_n_steps=50,
    #                   callbacks=[lr_monitor, checkpoint_callback], limit_train_batches=100, limit_val_batches=100)
    # datacls = AgricultureDataClass(DATASET_ROOT, DATASET_NAME, batch_size=10)
    # model = AgricultureSegmentClass(MODEL_NAME, NUM_CLASSES)

    cli = LightningCLI(save_config_callback=None, seed_everything_default=2573406166)
    # trainer.fit(model, datacls)
    # trainer.save_checkpoint("best_model.ckpt")
