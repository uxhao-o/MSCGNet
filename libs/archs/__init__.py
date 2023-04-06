#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: __init__.py.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 13:39

def load_model(name='MSCG-Rx50', classes=7, node_size=(32,32)):
    if name == 'MSCG-Rx50':
        net = rx50_gcn_3head_4channel(out_channels=classes)
    elif name == 'MSCG-Rx101':
        net = rx101_gcn_3head_4channel(out_channels=classes)
    else:
        print('not found the net')
        return -1

    return net
