#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: data_class.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 14:08
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from libs.data.dataset import split_train_val_test_sets, AgricultureDataset


class AgricultureDataClass(LightningDataModule):
    def __init__(self, dataset_root, data_name, input_size=(512, 512), batch_size=10, num_workers=4, k_folder=0,
                 kf=0, pre_norm=False, scale_rate=1.):
        super(AgricultureDataClass, self).__init__()
        self.dataset_root = dataset_root
        self.data_name = data_name
        self.train_root = os.path.join(dataset_root, "train")
        self.val_root = os.path.join(dataset_root, "val")
        self.test_root = os.path.join(dataset_root, "test")
        self.bands = ['NIR', 'RGB']
        self.k = kf
        self.k_folder = k_folder
        self.seeds = 69278
        self.pre_norm = pre_norm  # 是否归一化到[-1, 1]
        assert isinstance(input_size, tuple)
        self.input_size = input_size
        self.scale_rate = scale_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        train_dict, val_dict, test_dict = split_train_val_test_sets(dataset_root=self.dataset_root,
                                                                    train_root=self.train_root,
                                                                    name=self.data_name,
                                                                    val_root=self.val_root, KF=self.k_folder,
                                                                    k=self.k, seeds=self.seeds)
        self.train_set = AgricultureDataset(mode='train', file_lists=train_dict, pre_norm=self.pre_norm,
                                       winSize=self.input_size, scale=self.scale_rate)
        self.val_set = AgricultureDataset(mode='val', file_lists=val_dict, pre_norm=self.pre_norm,
                                     winSize=self.input_size, scale=self.scale_rate)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader


if __name__ == '__main__':
    DATASET_ROOT = 'D:/Agriculture-Vision'
    dataset_name = 'Agriculture'
    a = AgricultureDataClass(dataset_root=DATASET_ROOT, data_name=dataset_name)
    a.setup('1')
    b = a.val_dataloader()  # [images, labels]
    c = a.train_dataloader()
    for i in c:
        print(len(i))
        print(i[0].shape)
        break
