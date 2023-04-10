#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: m.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 23:34
import numpy as np
import torch
import torchmetrics.functional as f
olderr = np.seterr(all='ignore')
from sklearn.metrics import confusion_matrix


def evaluate(preds, gts, num_classes):
    """
    关于混淆矩阵的计算参考https://mp.weixin.qq.com/s/6wB6-DuDrowywxw124oh3g
    :param val_outputs_gt:
    :param num_classes:
    :return:
    """
    conmatrix = f.confusion_matrix(preds, gts, task="multiclass", num_classes=num_classes, ignore_index=255)
    M, N = conmatrix.shape
    # 以下三个均是行向量
    tp = torch.zeros(M, dtype=torch.long)
    fp = torch.zeros(M, dtype=torch.long)
    fn = torch.zeros(M, dtype=torch.long)

    for i in range(M):
        tp[i] = conmatrix[i, i]  # 混淆矩阵对角线元素表示把第i类预测为i类的样本总数，即第i类的真正例TP
        fp[i] = torch.sum(conmatrix[:, i]) - tp[i]  # 混淆矩阵的每行的总和 减去 对角线元素， 即得到将其他类预测为i类的样本总数（假正例FP）
        fn[i] = torch.sum(conmatrix[i, :]) - tp[i]  # 混淆矩阵的每列的总和 减去 对角线元素， 即得到将i类预测为其他类别的样本总数（假负例FN）

    precision = tp / (tp + fp)  # = tp/col_sum [p0, p1,..., p6]
    recall = tp / (tp + fn)  # [r0, r1,..., r6]

    # f1分数
    f1_score = 2 * recall * precision / (recall + precision)  # f1_score

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix

    # 准确率
    acc = torch.diag(conmatrix).sum() / conmatrix.sum()

    # 每一类的准确率
    acc_cls = torch.diag(conmatrix) / conmatrix.sum(axis=ax_p)
    acc_cls = torch.nanmean(acc_cls)

    iou = tp / (tp + fp + fn)
    # np.nanmean 表示 求均值时当向量中出现nan时，则忽略nan项，（注意，不是置为0，是直接忽略）
    mean_iou = torch.nanmean(iou)

    # 加权准确率
    freq = conmatrix.sum(axis=ax_p) / conmatrix.sum()
    fwavacc = (freq[freq > torch.tensor(0)] * iou[freq > torch.tensor(0)]).sum()

    # 注意这里返回每个类别F1分数的均值的原因是，我们希望每个类别的F1分数都要大，那么这些F1分数的均值也要大，因此可以使用每个类别F1分数的均值
    return acc, acc_cls, mean_iou, fwavacc, torch.nanmean(f1_score), conmatrix


def e1(preds, gts, num_classes):
    """
    关于混淆矩阵的计算参考https://mp.weixin.qq.com/s/6wB6-DuDrowywxw124oh3g
    :param gts:
    :param preds:
    :param num_classes:
    :return:
    """
    conmatrix = f.confusion_matrix(preds, gts, task="multiclass", num_classes=num_classes, ignore_index=255)
    M, N = conmatrix.shape
    # 以下三个均是行向量
    tp = torch.zeros(M, dtype=torch.long).cuda()
    fp = torch.zeros(M, dtype=torch.long).cuda()
    fn = torch.zeros(M, dtype=torch.long).cuda()

    for i in range(M):
        tp[i] = conmatrix[i, i]  # 混淆矩阵对角线元素表示把第i类预测为i类的样本总数，即第i类的真正例TP
        fp[i] = torch.sum(conmatrix[:, i]) - tp[i]  # 混淆矩阵的每行的总和 减去 对角线元素， 即得到将其他类预测为i类的样本总数（假正例FP）
        fn[i] = torch.sum(conmatrix[i, :]) - tp[i]  # 混淆矩阵的每列的总和 减去 对角线元素， 即得到将i类预测为其他类别的样本总数（假负例FN）

    precision = tp / (tp + fp)  # = tp/col_sum [p0, p1,..., p6]
    recall = tp / (tp + fn)  # [r0, r1,..., r6]

    # f1分数
    f1_score = 2 * recall * precision / (recall + precision)  # f1_score

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix

    # 准确率
    acc = torch.diag(conmatrix).sum() / conmatrix.sum()

    # 每一类的准确率
    acc_cls = torch.diag(conmatrix) / conmatrix.sum(axis=ax_p)
    acc_cls = torch.nanmean(acc_cls)

    iou = tp / (tp + fp + fn)
    # np.nanmean 表示 求均值时当向量中出现nan时，则忽略nan项，（注意，不是置为0，是直接忽略）
    mean_iou = torch.nanmean(iou)

    # 加权准确率
    freq = conmatrix.sum(axis=ax_p) / conmatrix.sum()
    fwavacc = (freq[freq > torch.tensor(0).cuda()] * iou[freq > torch.tensor(0).cuda()]).sum()

    # 注意这里返回每个类别F1分数的均值的原因是，我们希望每个类别的F1分数都要大，那么这些F1分数的均值也要大，因此可以使用每个类别F1分数的均值
    return acc, acc_cls, mean_iou, fwavacc, torch.nanmean(f1_score), conmatrix