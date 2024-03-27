
# for inference only

import argparse
import os
from core.predictor_classification import Predictor
import torch
import yaml
print('gpu check available', torch.cuda.is_available())
print('total device count', torch.cuda.device_count())
# set current device
torch.cuda.set_device(0) # 1 or 0
print('current device', torch.cuda.current_device())

import json

cfg = './conf/config_inference2_classification.yaml'
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt  # plotting tools

conf = OmegaConf.load(cfg)

print(OmegaConf.to_yaml(conf))

def set_state(idd, cfg):
    with open(cfg) as f:
        doc = yaml.safe_load(f)
    old = doc['input_image_dir'][-5:]
    doc['input_image_dir'] = doc['input_image_dir'].replace(old, idd)
    doc['output_dir'] = doc['output_dir'].replace(old, idd)
    print(doc['input_image_dir'])
    with open(cfg, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=0)
    return


conf = OmegaConf.load(cfg)

# print(OmegaConf.to_yaml(conf))
pred = Predictor(conf)
c, notwork, nochm = pred.predict_all()
dicc = {'notworking files': notwork, 'nochm files': nochm}
a_file = open(conf.output_dir + '/pred_summary.json', 'w+')
json.dump(dicc, a_file)
a_file.close()