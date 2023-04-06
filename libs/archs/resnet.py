#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: resnet.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 22:05
import torch
from torch import nn
import torch.nn.functional as F
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d

from libs.archs.scg_gcn import GCN_Layer, SCG_block, weight_xavier_init

__all__ = ['MSCG_Rx50']


class RX50Gcn3Head4channel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(RX50Gcn3Head4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = num_classes

        resnet = se_resnext50_32x4d()
        # 读取预训练resnet50的bottleneck层
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        # 因为数据集是NIR-RGB图像，有4个通道，所以需要重新设置backbone的第1个bottleneck层
        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 提取backbone的第1个卷积层的权重参数
        for child in self.layer0.children():  # 提取第1个bottleneck层的第一个卷积层。
            for param in child.parameters():
                par = param
                break
            break

        # 复制卷积层红色通道的参数， par[滤波器组数, R, G, B]
        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)  # 在原来backbone 3个通道卷积上增加1个通道变成4个通道
        # backbone的新的第1个bottleneck层
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, num_classes, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=1024,
                             hidden_ch=num_classes,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)
        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)
        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.reshape(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:  # 训练模式下自动为True
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:  # 验证模式
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)


def MSCG_Rx50(num_classes):
    return RX50Gcn3Head4channel(num_classes)
