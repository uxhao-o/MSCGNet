#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: dataset.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 13:55
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
from libs.data.utils import labels_folder, check_mkdir, img_basename
from libs.data.augment import *
from sklearn.model_selection import train_test_split, KFold


IMG = 'images' # RGB or IRRG, rgb/nir
GT = 'gt'
IDS = 'IDs'


def prepare_gt(root_folder, out_path='gt'):
    if not os.path.exists(os.path.join(root_folder, out_path)):
        print('----------creating groundtruth data for training./.val---------------')
        check_mkdir(os.path.join(root_folder, out_path))
        basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'images/rgb'))]
        gt = basname[0]+'.png'
        for fname in basname:
            gtz = np.zeros((512, 512), dtype=int)
            for key in labels_folder.keys():
                # 将同一张图片的每个语义gt合并为一个gt
                gt = fname + '.png'
                mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, gt), -1) / 255, dtype=int) * labels_folder[key]
                gtz[gtz < 1] = mask[gtz < 1]

            for key in ['boundaries', 'masks']:
                mask = np.array(cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int)
                gtz[mask == 0] = 255

            cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def get_data_folder(DATASET_ROOT):
    """
    用于多数据集加载
    :param DATASET_ROOT:
    :return:
    """
    Data_Folder = {
        'Agriculture': {
            'ROOT': DATASET_ROOT,
            'RGB': 'images/rgb/{}.jpg',
            'NIR': 'images/nir/{}.jpg',
            'SHAPE': (512, 512),
            'GT': 'gt/{}.png',
        },
    }
    return Data_Folder


def get_training_list(data_root, count_label=True):
    """
    统计每个语义类别下的样本 和 训练集样本名称列表
    :param data_root:
    :param count_label:
    :return:
    """
    dict_list = {}
    # os.path.splitext 分离文件名与扩展名 ('11T3V93AF_2280-3416-2792-3928', '.jpg')
    # 获取所有图片样本的名称，不包括后缀名
    basename = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_root, 'images/nir'))]
    # 统计每个语义类别下的样本
    if count_label:
        # 遍历读取每个语义标签下的所有图的gt, 合并为一个列表
        for key in labels_folder.keys():  # key是语义标签名称
            no_zero_files = []
            for fname in basename:
                # 读取该语义标签下的所有图像的gt
                # IMREAD_UNCHANGED 读入完整图片包括alpha通道，
                # alpha通道每个像素点的取值为[0,1], 取值为0时代表该像素点对图像没有贡献，取值为1时代表该像素点对图像有贡献
                gt = np.array(cv2.imread(os.path.join(data_root, 'labels', key, fname+'.png'), cv2.IMREAD_UNCHANGED))  # (512,512)
                if np.count_nonzero(gt):  # 若这张图像包含这类语义, 则加入list
                    # np.count_nonzero 统计矩阵中的非0像素点个数
                    no_zero_files.append(fname)  # 将包含该类语义的样本加入到列表中
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basename


def split_train_val_test_sets(dataset_root, train_root, val_root, name='Agriculture', bands=['NIR', 'RGB'], KF=3, k=1, seeds=69278):
    """

    :param dataset_root:
    :param train_root: 训练集根路径
    :param val_root: 验证集根路径
    :param name: 使用的数据集名
    :param bands:
    :param KF: 划分为多少折
    :param k:
    :param seeds: 随机因子
    :return:
    """
    train_id, train_list = get_training_list(data_root=train_root, count_label=False)
    val_id, val_list = get_training_list(data_root=val_root, count_label=False)

    if KF >= 2:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)  # k折交叉验证器
        val_ids = np.array(train_list)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        t2_list, v_list = {list(val_ids[idx[k][0]]), list(val_ids[idx[k][1]])}
    else:
        t2_list = []

    data_folder = get_data_folder(dataset_root)

    # 读取每张nir和RGB两种格式， 即一个样本有2张图，可看成数据扩充。（或者说nir和RGB都用来训练）
    # list[img_nir, img_rgb]
    img_folders = [os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name][band]) for band in bands]
    gt_folder = os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name]['GT'])

    # list[img_nir, img_rgb]
    val_folders = [os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name][band]) for band in bands]
    val_gt_folder = os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])

    train_dict = {
        'IDs': train_id,
        'images': [[img_folder.format(id) for img_folder in img_folders] for id in train_list] +
             [[val_folder.format(id) for val_folder in val_folders] for id in t2_list],  # 合并两个二维列表，将后一个的所有行追加到第一个的最后
        'gt': [gt_folder.format(id) for id in train_list] + [val_gt_folder.format(id) for id in t2_list],
        'all_files': train_list + t2_list
    }

    val_dict = {
        'IDs': val_id,
        'images': [[val_folder.format(id) for val_folder in val_folders] for id in val_list],
        'gt': [val_gt_folder.format(id) for id in val_list],
        'all_files': val_list
    }

    test_dict = {
        'IDs': val_id,
        'images': [[val_folder.format(id) for val_folder in val_folders] for id in val_list],
        'gt': [val_gt_folder.format(id) for id in val_list],
    }

    print('train set -------', len(train_dict['gt']))
    print('val set ---------', len(val_dict['gt']))
    return train_dict, val_dict, test_dict


class AgricultureDataset(Dataset):
    def __init__(self, mode="train", file_lists=None, winSize=(256, 256), pre_norm=False, scale=1.0/1.0):
        super(AgricultureDataset, self).__init__()
        # assert断言机制，若其后的条件未Ture时正常执行，为False立即触发异常，不必等待程序运行后出现崩溃的情况
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.norm = pre_norm
        self.winsize = winSize
        self.scale = scale
        self.all_ids = file_lists['all_files']
        self.image_files = file_lists['images']  # image_files = [[bands1, bands2,..], ...]
        self.mask_files = file_lists['gt']  # mask_files = [gt1, gt2, ...]

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):

        if len(self.image_files) > 1:
            imgs = []
            for k in range(len(self.image_files[idx])):
                filename = self.image_files[idx][k]
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    img = imload(filename, gray=True, scale_rate=self.scale)
                    img = np.expand_dims(img, 2)
                    imgs.append(img)
                else:
                    img = imload(filename, scale_rate=self.scale)
                    imgs.append(img)
            image = np.concatenate(imgs, 2)
        else:
            filename = self.image_files[idx][0]
            path, _ = os.path.split(filename)
            if path[-3:] == 'nir':
                image = imload(filename, gray=True, scale_rate=self.scale)
                image = np.expand_dims(image, 2)
            else:
                image = imload(filename, scale_rate=self.scale)

        label = imload(self.mask_files[idx], gray=True, scale_rate=self.scale)

        if self.winsize != label.shape:
            image, label = img_mask_crop(image=image, mask=label,
                                         size=self.winsize, limits=self.winsize)

        if self.mode == 'train':
            image_p, label_p = self.train_augmentation(image, label)
        elif self.mode == 'val':
            image_p, label_p = self.val_augmentation(image, label)

        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0  # [H, W, C] -> [C, H, W]
        label_p = np.asarray(label_p, dtype='int64')

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            # MedianBlur(p=0.2),
            # Transpose(p=0.5),
            # RandomSizedCrop(min_max_height=(128, 512), height=512, width=512, p=0.1),
            # ShiftScaleRotate(p=0.2,
            #                  rotate_limit=10, scale_limit=0.1),
            # ChannelShuffle(p=0.1),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def val_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def normalize(cls, img):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        norm = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return norm(img)


if __name__ == '__main__':
    a, b = get_training_list()
    print(a)
