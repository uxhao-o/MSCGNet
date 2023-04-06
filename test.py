#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: test.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 13:42
import re
from typing import Dict

from libs.archs import resnet

a = "{epoch:02d}-{val_loss:.2f}-{acc:.2f}-{acc_cls:.2f}-{miou:.2f}-{f1:.2f}"
groups = re.findall(r"(\{.*?)[:\}]", a)

for group in groups:
    name = group[1:]

    # if auto_insert_metric_name:
    #     filename = filename.replace(group, name + "={" + name)

    # # support for dots: https://stackoverflow.com/a/7934969
    # filename = filename.replace(group, f"{{0[{name}]")
    print(name)

print(groups)
b = {
    'epoch': 0.1,
    'val_loss': 0.2,
    'acc': 0.01,
    'acc_cls': 0.21,
    'miou': 0.67,
    'f1': 0.22
}

print(a.format(b))
# print(resnet.__dict__['MSCG_Rx50'](out_channels=7))
# print(resnet.__dict__.keys())

# import numpy as np
#
#
# a = np.zeros(shape=(2, 3, 4))
# b = []
# for i in range(3):
#     b.append(a)
#
# c = np.array(b)
#
# print(len(c.flatten()))

import pytorch_lightning as pl

print(pl.__version__)