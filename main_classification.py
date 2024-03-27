#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:13:37 2022

@author: sizhuo
"""

import argparse
import os
from core.trainer import Trainer
import glob
import ipdb
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # first gpu # not working well
import torch
import yaml
print('gpu check available', torch.cuda.is_available())
print('total device count', torch.cuda.device_count())
# set current device
torch.cuda.set_device(0) # 1 or 0
print('current device', torch.cuda.current_device())

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

cfg = './conf/config_classification.yaml'
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt  # plotting tools
import wandb
# conf = OmegaConf.load(cfg)

# print(OmegaConf.to_yaml(conf))

# Trainer(conf, showImOnly =0)




model_base = [
    'torchEfficientnetb0',
    'torchEfficientnetb1',
    'torchEfficientnetb2',
    ]



search_list = model_base



def set_state(mm, cfg, addchm = 0):
    with open(cfg) as f:
        doc = yaml.safe_load(f)

    doc['model_type'] = mm


    with open(cfg, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=0)
    return


for c in range(len(search_list)):


    set_state(search_list[c], cfg)


    conf = OmegaConf.load(cfg)

    print(OmegaConf.to_yaml(conf))

    Trainer(conf, showImOnly = 0)

    print('================================================================')


print('finished grid search')
print('-----------------------')
