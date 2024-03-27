#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:27:20 2022

@author: sizhuo
"""

from torch.backends import cudnn
import random
import matplotlib.pyplot as plt  # plotting tools
import numpy as np
import glob
import json
import time
import pandas as pd
import os
from tqdm import tqdm
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import rasterio
from core.evaluation import *
from core.data_loader_classification import get_loader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import csv
from torchinfo import summary
from itertools import product
from rasterio import windows
from torchvision import transforms as T
from rasterio.enums import Resampling
from scipy.ndimage import zoom
import ipdb
import torchvision.models as models


class Predictor:
    # inference only
    def __init__(self, config):
        self.config = config
        self.all_files = load_files(config)
        self.model_type = config.model_type
        self.pretrained = config.pretrained
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weightDecay = config.weightDecay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # print(self.config)
        # load model and compile
        self.build_model()
        self.load_weights()

        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def build_model(self):
        """Build generator and discriminator."""
        if 'torchEfficientnetb0' in self.model_type:
            # self.nnet = models.efficientnet_b0(pretrained=self.pretrained, num_classes=self.output_ch)
            if not self.pretrained:
                self.nnet = models.efficientnet_b0(num_classes=self.output_ch)
                self.nnet.features[0][0] = nn.Conv2d(self.img_ch, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                # self.nnet.features[2][0].block[1][0] = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
                self.nnet.features[-1][0] = nn.Conv2d(320, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.nnet.features[-1][1] = nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.nnet.classifier[1] = nn.Linear(in_features=480, out_features=1, bias=True)

            else: # load pretrained, only reini the final layer
                if 'dense' in self.model_type:
                    # load pretrained, only reini the final layer
                    # net0 = models.efficientnet_b0(pretrained=True)
                    net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                    num_fs = net0.classifier[1].in_features

                    self.nnet = net0
                    self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=640, out_features=320, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=320, out_features=64, bias=True),
                                                        nn.Dropout(p=0.2),
                                                        nn.Linear(in_features=64, out_features=1, bias=True))


                elif 'map' in self.model_type:
                    net0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

                    # shrink regression output
                    self.nnet = net0
                    self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                    self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                    # self.nnet.features.append(nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    # self.nnet.features[-1][1] = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

                    # remove avg poly
                    self.nnet.avgpool = torch.nn.Identity()

                    self.nnet.classifier = torch.nn.Sequential()



        elif 'torchEfficientnetb1' in self.model_type:
            print('model architecture loaded for ', self.model_type)
            assert self.pretrained == 1
            self.nnet = models.efficientnet_b1(pretrained=self.pretrained)
            if 'dense' in self.model_type:
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=640, out_features=320, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=320, out_features=64, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=64, out_features=1, bias=True))


            elif 'map' in self.model_type:
                self.nnet.features[-1][0] = nn.Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                self.nnet.classifier = torch.nn.Sequential()


        elif 'torchEfficientnetb2' in self.model_type:
            assert self.pretrained == 1
            print('model architecture loaded for ', self.model_type)
            self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
            if 'dense' in self.model_type:
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=640, out_features=320, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=320, out_features=64, bias=True),
                                                    nn.Dropout(p=0.2),
                                                    nn.Linear(in_features=64, out_features=1, bias=True))

            elif 'map' in self.model_type:
                # ipdb.set_trace()
                num_fs = self.nnet.features[-2][-1].block[-1][0].out_channels
                self.nnet.features[-1][0] = nn.Conv2d(num_fs, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.nnet.features[-1][1] = nn.BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.nnet.features.append(nn.Conv2d(640, 240, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False))
                self.nnet.features.append(nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                self.nnet.features.append(nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True))
                self.nnet.avgpool = torch.nn.Identity()

                self.nnet.classifier = torch.nn.Sequential()


        if not self.pretrained:

            self.optimizer = optim.Adam(list(self.nnet.parameters()),
                                      self.lr, [self.beta1, self.beta2], self.weightDecay)

        else: # using pretrained model
            params_to_update = self.nnet.parameters()
            print("Params to learn:")

            params_to_update = []
            for name,param in self.nnet.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)


            # Observe that all parameters are being optimized
            self.optimizer = optim.Adam(params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay)



        self.nnet.to(self.device)

        self.print_network(self.nnet, self.model_type)




    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # import ipdb
        # ipdb.set_trace()

        print('------Model name------- ', name)

        model_stats = summary(model, input_size=[self.config.BATCH_SIZE, self.config.img_ch, self.config.inputlength, self.config.inputlength], col_names = ("input_size", "output_size", "num_params"))
        # model_stats = summary(model, input_size=(32, 3, 180, 180))
        print(model_stats)
        print("The number of parameters: {}".format(num_params))
        self.summary_str = str(model_stats)


    def load_weights(self):
        if os.path.isfile(self.config.model_path):
            # Load the pretrained Encoder
            self.nnet.load_state_dict(torch.load(self.config.model_path))
            print('%s is Successfully Loaded from %s'%(self.config.model_type,self.config.model_path))
        # self.unet.load_state_dict(torch.load(self.model_path))

        self.nnet.train(False)
        self.nnet.eval()


    def predict_all(self):
        c, notwork, nochm = predict_labels(self.config, self.all_files, self.nnet, self.device)
        return c, notwork, nochm



def load_files(config):
    all_files0 = glob.glob(f"{config.input_image_dir}/**/{config.input_image_pref}*{config.input_image_type}", recursive=True)
    all_files = [(i, os.path.basename(i)) for i in all_files0]
    print('**********************************')
    print('Number of raw image to predict:', len(all_files))
    return all_files


def predict_labels(config, all_files, model, device):
    counter = 1
    notwork = []
    nochm = []
    outputFiles = []
    outputFile = os.path.join(config.output_dir, config.output_suffix + config.output_table_type)
    if not os.path.exists(outputFile):
        outputFiles.append(outputFile)
        pred_loader = get_loader(image_path=config.input_image_dir, config = config,
                                split_list = [],
                                mode='pred')
#         ipdb.set_trace()
        pd_list = []
        for i, data in tqdm(enumerate(pred_loader)):
            if i%10:
                print(i)
            images = data

            images = images.to(device)

            pred = model(images).squeeze()
            if config.task == 'classification':
                pred = torch.sigmoid(pred)
            pred = pred.cpu().detach().numpy()
            
#             ipdb.set_trace()
            pd_list.extend(pred)
        df = pd.DataFrame(list(zip(all_files, pd_list)),
                                      columns =['name','pred'])
        df.to_csv(outputFile)
        counter += 1

    else:
        print('Skipping: File already analysed!')

    return counter, notwork, nochm



def rgb2gray(rgb):

    r, g, b = rgb[:, :,0], rgb[:, :,1], rgb[:, :,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
