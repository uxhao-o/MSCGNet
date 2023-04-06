#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: preprocess.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: 数据预处理
# Date: 2023/3/31 13:46
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold


palette_land = {
    0: (0, 0, 0),        # background
    1: (255, 255, 0),    # cloud_shadow
    2: (255, 0, 255),    # double_plant
    3: (0, 255, 0),      # planter_skip
    4: (0, 0, 255),      # standing_water
    5: (255, 255, 255),  # waterway
    6: (0, 255, 255),    # weed_cluster
}


# 自定义调色板，便于可视化，便于论文阅读
palette_vsl = {
    0: (0, 0, 0),     # background
    1: (0, 255, 0),     # cloud_shadow
    2: (255, 0, 0),     # double_plant
    3: (0, 200, 200),   # planter_skip
    4: (255, 255, 255),  # standing_water
    5: (128, 128, 0),   # waterway
    6: (0, 0, 255)        # weed_cluster
}

labels_folder = {
    'cloud_shadow': 1,
    'double_plant': 2,
    'planter_skip': 3,
    'standing_water': 4,
    'waterway': 5,
    'weed_cluster': 6
}

# 7个类别（包含背景）
land_classes = ["background", "cloud_shadow", "double_plant", "planter_skip",
                "standing_water", "waterway", "weed_cluster"]
