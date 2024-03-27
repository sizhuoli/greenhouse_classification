#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:45:21 2022

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

cfg = './conf/config_testscore.yaml'
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt  # plotting tools
import wandb
conf = OmegaConf.load(cfg)
import pandas as pd
# search for all models saved in the folder and report test scores
modelps = glob.glob(f'{conf.model_path_base}/*/')


models = []
for metric in ['F1','Loss']:
    models.extend([f + 'best{}.pkl'.format(metric) for f in modelps])
    
print('Models for evaluation:')
print(models)

def set_state(mm, cfg):
    with open(cfg) as f:
        doc = yaml.safe_load(f)

    doc['model_path'] = mm      

    with open(cfg, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=0)
    return doc['model_type']


names = ['model', 'save_metric', 'f1', 'acc', 'recall', 'precision']
scores = {k: [] for k in names}

for c in range(len(models)):
    
            
    mm_type = set_state(models[c], cfg)
            
    conf = OmegaConf.load(cfg)
    
    print(OmegaConf.to_yaml(conf))

    labels, preds, f1_score, acc, recall, precision = Trainer(conf, showImOnly = 0)
    
    if 'bestF1' in models[c]:
        save_m = 'F1' 
    elif 'bestLoss' in models[c]:
        save_m = 'focal'
    
    values = [mm_type, save_m, f1_score, acc, recall, precision]
    
    for i in range(len(names)):
        scores[names[i]].append(values[i])
    
    print('================================================================')



dd2 = pd.DataFrame.from_dict(scores)
print(dd2)
dd2.to_csv('/mnt/super/greenhouse/validation/test_scores/scores.csv')

