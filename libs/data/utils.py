#!/usr/bin/python3
# -*- coding:utf-8 -*-
# File: utils.py
# Author: uxhao
# Contact: uxhao_o@163.com
# Description: TODO
# Date: 2023/3/31 14:46
import os


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def img_basename(filename):
    """
    获取image
    :param filename:
    :return:
    """
    # 分离文件名与扩展名 ('11T3V93AF_2280-3416-2792-3928', '.jpg')
    return os.path.basename(os.path.splitext(filename)[0])


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg'])


palette_land = {
    0: (0, 0, 0),        # background
    1: (255, 255, 0),    # cloud_shadow
    2: (255, 0, 255),    # double_plant
    3: (0, 255, 0),      # planter_skip
    4: (0, 0, 255),      # standing_water
    5: (255, 255, 255),  # waterway
    6: (0, 255, 255),    # weed_cluster
}

# customised palette for visualization, easier for reading in paper
palette_vsl = {
    0: (0, 0, 0),     # background
    1: (0, 255, 0),     # cloud_shadow
    2: (255, 0, 0),     # double_plant
    3: (0, 200, 200),   # planter_skip
    4: (255, 255, 255), # standing_water
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

land_classes = ["background", "cloud_shadow", "double_plant", "planter_skip",
                "standing_water", "waterway", "weed_cluster"]




IMG = 'images' # RGB or IRRG, rgb/nir
GT = 'gt'
IDS = 'IDs'