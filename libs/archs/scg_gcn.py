#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: scg_gcn.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 22:07
import torch
import torch.nn as nn


class SCG_block(nn.Module):
    def __init__(self, in_ch, hidden_ch=6, node_size=(32, 32), add_diag=True, dropout=0.2):
        super(SCG_block, self).__init__()
        self.node_size = node_size
        self.hidden = hidden_ch
        self.nodes = node_size[0] * node_size[1]
        self.add_diag = add_diag
        self.pool = nn.AdaptiveAvgPool2d(node_size)

        # 3×3 conv, 为了产生均值矩阵M
        self.mu = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=True),
            nn.Dropout(dropout),
        )
        # 1×1 conv, 为了产生对数标准差矩阵log(Σ) shape(h*w, c) c表示语义的数目
        # 对Σ取对数是为了保证训练稳定 和 Σ的所有元素为正值
        self.logvar = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, 1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.size()  # 批量,通道,高,宽

        # 对SCG输入F采用自适应平均池化（AdaptiveAvgPool2d）降低空间维度，得到F_
        gx = self.pool(x)  # F shape(h, w, d) -> F_ shape(h_, w_, d)

        # 为了产生均值矩阵M 和 对数标准差矩阵log(Σ) 分别对F_ 使用conv3×3 和 conv1×1
        # M shape(B, c, h_, w_), log(Σ) shape(B, c, h_, w_)
        mu, log_var = self.mu(gx), self.logvar(gx)

        if self.training:  # 原始SCG, Z = M + Σ·r
            std = torch.exp(log_var.reshape(B, self.nodes, self.hidden))  # shape(B, h_*w_, c)
            # torch.randn_like()，返回与输入相同大小的张量，该张量由均值为0和方差为1的正态分布中的随机数填充。
            eps = torch.randn_like(std)  # 辅助噪声r shape(h_*w_, c)
            # (B, h_*w_, c) + (B, h_*w_, c)
            z = mu.reshape(B, self.nodes,
                           self.hidden) + std * eps  # * 与 mul 等价，表示对应位置元素相乘，当两个张量维度不同时，通过广播操作将相乘的两个张量的维度变得相同。
        else:  # SCG_ae, 自动编码模块下 Z=Flatten(conv_3×3(F_))
            # Z (B, h_*w_, c)
            z = mu.reshape(B, self.nodes, self.hidden)

        # 邻接权重矩阵 A=relu(ZZ^T)
        A = torch.matmul(z, z.permute(0, 2, 1))
        A = torch.relu(A)  # (B, h_*w_, h_*w_)

        # 对角对数正则项L_dl
        # torch.diagonal 若目标是二维张量，即取对角线元素；若目标是三维张量，即代表分别取B个形状为(h_*w_)×(h_*w_)的张量的对角线元素
        # Ad shape(B, h_*w_)
        Ad = torch.diagonal(A, dim1=1, dim2=2)  # 分别取B个形状为(h_*w_)×(h_*w_)的张量的对角线元素，每个张量的对角线元素依次组成一个行向量，最终返回含有B个行向量的矩阵
        # 求Ad矩阵每行的均值，即每个邻接矩阵对角线元素的均值，并压缩元素值到[0,1]之间
        mean = torch.mean(Ad, dim=1).clamp(min=0.001)  # or mean = mean + 1.e-3

        # 适应性因子γ
        gama = torch.sqrt(1 + 1.0 / mean).unsqueeze(-1).unsqueeze(-1)

        # 对角线对数损失函数
        dl_loss = gama.mean() * torch.log(Ad[Ad < 1] + 1.e-7).sum() / (A.size(0) * A.size(1) * A.size(2))
        # KL散度损失函数
        kl_loss = -0.5 / self.nodes * torch.mean(
            torch.sum(1 + 2 * log_var - mu.pow(2) - log_var.exp().pow(2), 1)
        )

        loss = kl_loss - dl_loss

        if self.add_diag:  # 使用适用于邻接矩阵和残差嵌入的自适应增强方法。
            diag = []  # 存储B个对角矩阵
            for i in range(Ad.shape[0]):
                '''
                torch.diag(x):
                    如果 x 是向量（一维张量），则返回以 x 的元素为对角线的矩阵。
                    如果 x 是矩阵（二维张量），则返回具有 x 对角线元素的一维张量。
                '''
                # 将Ad (B, h_*w_) 的每一行的元素作为对角线元素，生成对角矩阵
                diag.append(torch.diag(Ad[i, :]).unsqueeze(0))  # 返回B行数据各自的对角矩阵

            # 自适应增强的邻接矩阵A
            A = A + gama * torch.cat(diag, 0)  # A = A+γ·d
            # A = A + A * (gama * torch.eye(A.size(-1), device=A.device).unsqueeze(0))

        # A = laplacian_matrix(A, self_loop=True)
        A = self.laplacian_matrix(A, self_loop=True)
        # A = laplacian_batch(A.unsqueeze(3), True).squeeze()

        z_hat = gama.mean() * \
                mu.reshape(B, self.nodes, self.hidden) * \
                (1. - log_var.reshape(B, self.nodes, self.hidden))

        return A, gx, loss, z_hat

    @classmethod
    def laplacian_matrix(cls, A, self_loop=False):
        '''
        Computes normalized Laplacian matrix: A (B, N, N)
        '''
        if self_loop:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)
        # deg_inv_sqrt = (A + 1e-5).sum(dim=1).clamp(min=0.001).pow(-0.5)
        deg_inv_sqrt = (torch.sum(A, 1) + 1e-5).pow(-0.5)

        LA = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

        return LA


class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, bnorm=True,
                 activation=nn.ReLU(), dropout=None):
        super(GCN_Layer, self).__init__()
        self.bnorm = bnorm
        fc = [nn.Linear(in_features, out_features)]
        if bnorm:
            fc.append(BatchNorm_GCN(out_features))
        if activation is not None:
            fc.append(activation)
        if dropout is not None:
            fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x, A = data
        y = self.fc(torch.bmm(A, x))

        return [y, A]


def weight_xavier_init(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.xavier_normal_(module.weight)
                nn.init.orthogonal_(module.weight)
                # nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class BatchNorm_GCN(nn.BatchNorm1d):
    '''Batch normalization over GCN features'''

    def __init__(self, num_features):
        super(BatchNorm_GCN, self).__init__(num_features)

    def forward(self, x):
        return super(BatchNorm_GCN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)
