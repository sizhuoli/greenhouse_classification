#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:27:20 2022

@author: sizhuo
"""

from core.solver_new import Solver
from core.data_loader_classification import get_loader
from torch.backends import cudnn
import random
import matplotlib.pyplot as plt  # plotting tools
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import json
import time
import pandas as pd
import os
import ipdb

def Trainer(config, showImOnly = 0, launch_wandb = 1):

    if config.mergeTrainValid:
        if config.task == 'classification':
            GT_dd = pd.read_csv(config.classification_label_fn)
            print('GTdd', len(GT_dd))
            selected_fns = GT_dd.loc[GT_dd['useCase']=='train', 'path'].to_list()
            print('sel len', len(set(selected_fns)))
            all_files = glob.glob(f'{config.train_path}/**/*.tif', recursive=True)
            print('all', len(all_files))
            image_paths = [f for f in all_files if f in selected_fns]
            print('sel', len(image_paths))
            train_list, valid_list = train_test_split(image_paths, test_size=1-config.split_ratio)
            print('train', len(train_list), len(valid_list))
            timestr = time.strftime("%Y%m%d-%H%M")
            json_base = config.train_path + 'trainValid_split/'
            if not os.path.exists(json_base):
                os.mkdir(json_base)
            json_file = json_base + 'split_' + timestr + '_split_' + str(int(config.split_ratio*10)) + '.json'
            data = {}
            data['training'] = train_list
            data['validation'] = valid_list
            data['num_train'] = len(train_list)
            data['num_valid'] = len(valid_list)
            data['train_ratio'] = config.split_ratio

            with open(json_file, 'w') as file:
                json.dump(data, file)

        else:
            print('task invalid')


    else:
        train_list = valid_list = []



    train_loader = get_loader(image_path=config.train_path, config = config,
                            split_list = train_list,
                            mode='train')

    valid_loader = get_loader(image_path=config.valid_path,
                              config = config,
                              split_list = valid_list,
                            mode='valid')

    test_loader = get_loader(image_path=config.test_path,
                            config = config,
                            split_list = train_list + valid_list,
                            mode='test')

    # import ipdb
    # ipdb.set_trace()
    if config.mode == 'train':
        print('Checking training data')

        images = data_vis(train_loader, config.notree, config.conf_score, config.use_color, config.add_chm, config.add_seg, twoinput = config.add_input2, outputs = config.add_outputs)

        print('Checking validation data')
        _ = data_vis(valid_loader, config.notree, config.conf_score, config.use_color, config.add_chm, config.add_seg, twoinput = config.add_input2, outputs = config.add_outputs)

    else:
        images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chm, config.add_seg, num = 1 , test_vis=config.test_vis)
        # images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chm, config.add_seg, num = 1, model = solver.nnet, devi = solver.device, test = 1)
        print('***')
    if config.add_input2:
        patch_size = [images[0].shape, images[1].shape]
    else:
        patch_size = images.shape
    if not showImOnly:
        # print('patch size:', patch_size)
        solver = Solver(config, train_loader, valid_loader, test_loader, patch_size, launch_wandb = launch_wandb)

        # Train and sample the images
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            labels, preds, f1_score, acc, recall, precision = solver.test()
        if config.mode == 'test':
            print('Checking testign data')
#             images = data_vis(test_loader, config.notree, config.conf_score, config.use_color, config.add_chm, config.add_seg, num = 1, model = solver.nnet, devi = solver.device, test = 1)
    if 'test' in config.mode:
        return labels, preds, f1_score, acc, recall, precision
    else:
        return

def data_vis(loader, notree, conf_score, use_color, add_chm, add_seg, num = 3, model = '', devi = '', test = 0, twoinput = 0, outputs = 0, test_vis = 0):
    for _ in range(num):
        dataiter = iter(loader)

        if not conf_score:
            # images, labels = dataiter.next()
            images, labels = next(dataiter)
        else:

            if twoinput:

                images, input2, labels, sp_wei = next(dataiter)

            elif outputs:
                images, labels, add_op1, add_op2, sp_wei = next(dataiter)
            elif test_vis:
                images, labels, sp_wei, fn = next(dataiter)
            else:
                images, labels, sp_wei = next(dataiter)

            sp_wei = sp_wei.numpy()
        labels = labels.numpy()
        if twoinput:
            input2 = input2.numpy()
        if outputs:
            add_op1, add_op2 = add_op1.numpy(), add_op2.numpy()
        print('lb', labels.shape)
        print(images.shape)
        print(images.mean())
        print(images.std())
        print(images.max(), images.min())
        print(images[0, 0, :50, :50])
        if test:
            model.train(False)
            model.eval()
            images = images.to(devi)
            pre = model(images)
            pre = pre.cpu().detach().numpy()
            images = images.cpu().detach()


        if not add_chm:
            if test:
                plt.figure(figsize = (20,20)) # size(width, height)
                for i in range(6):
                    for j in range(6): # column
                        plt.subplot(6, 6, 6*j + i+1)
                        curim = images[6*j + i].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                        # recover from meanstd
                        imstd = np.array([0.229, 0.224, 0.225])
                        immean = np.array([0.485, 0.456, 0.406])
                        curim = curim*imstd + immean
                        # print(curim.max(), curim.min())
                        plt.imshow(curim[:, :, :3])
                        # plt.title(str(labels[i]))
                        if notree:
                            plt.title([int(labels[6*j + i][0]), labels[6*j + i][1], labels[6*j + i][2]])
                        else:
                            plt.title(str(labels[6*j + i]))

                plt.tight_layout()
                # plt.show(block = False)
            else:
                plt.figure(figsize = (20,6)) # size(width, height)
                for i in range(6):

                    plt.subplot(1, 6, i+1)
                    curim = images[i].numpy()
                    curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                    # recover from meanstd
                    imstd = np.array([0.229, 0.224, 0.225])
                    immean = np.array([0.485, 0.456, 0.406])
                    curim = curim*imstd + immean
                    # print(curim.max(), curim.min())
                    # ipdb.set_trace()
                    if curim.dtype != 'int8':
                        curim = curim.astype('int8')
                    plt.imshow(curim[:, :, :3])
                    # plt.title(str(labels[i]))
                    if notree:
                        plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                    else:
                        plt.title(str(labels[i]))

                plt.tight_layout()
                # plt.show()


        elif add_chm:
            if use_color:
                plt.figure(figsize = (20,8)) # size(width, height)

                if test:
                    plt.figure(figsize = (20,20)) # size(width, height)
                    for i in range(6):
                        for j in range(6): # column
                            plt.subplot(6, 6, 6*j + i+1)
                            curim = images[6*j + i].numpy()
                            curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                            # recover from meanstd
                            imstd = np.array([0.229, 0.224, 0.225])
                            immean = np.array([0.485, 0.456, 0.406])
                            curim = curim*imstd + immean
                            # print(curim.max(), curim.min())
                            plt.imshow(curim[:, :, 0], cmap = 'gray')
                            # plt.title(str(labels[i]))
                            if notree:
                                plt.title([int(labels[6*j + i][0]), labels[6*j + i][1], labels[6*j + i][2]])
                            else:
                                plt.title(str(labels[6*j + i]))

                    plt.tight_layout()
                    # plt.show(block = False)

                else:

                    for i in range(6):

                        plt.subplot(3, 6, i+1)
                        curim = images[i].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        # recover from meanstd
                        imstd = np.array([0.229, 0.224, 0.225])
                        immean = np.array([0.485, 0.456, 0.406])
                        curim = curim*imstd + immean
                        # print(curim.shape)
                        # print(curim[:, :, :3].mean())
                        # import ipdb
                        # ipdb.set_trace()

                        plt.imshow(curim[:, :, 0].astype(float), cmap = 'gray')


                        if notree:
                            plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                        else:
                            plt.title(str(labels[i]))

                    for j in range(6):
                        # second row
                        # chm
                        plt.subplot(3, 6, j+7)
                        curim = images[j].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        imstd = np.array([0.229, 0.224, 0.225])
                        immean = np.array([0.485, 0.456, 0.406])
                        curim = curim*imstd + immean
                        plt.imshow(curim[:, :, 1], cmap = 'gray')
                        plt.colorbar()
                        # plt.title('max height: ' + str(round(curim[:, :, -1].max())))

                    for j in range(6):
                        # second row
                        # chm
                        plt.subplot(3, 6, j+13)
                        curim = images[j].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        imstd = np.array([0.229, 0.224, 0.225])
                        immean = np.array([0.485, 0.456, 0.406])
                        curim = curim*imstd + immean
                        plt.imshow(curim[:, :, -1], cmap = 'gray')
                        plt.colorbar()
                        # plt.title('max height: ' + str(round(curim[:, :, -1].max())))
                    # plt.show(block = False)
            else:
                if add_seg:
                    plt.figure(figsize = (20,8)) # size(width, height)
                    for i in range(6):

                        plt.subplot(2, 6, i+1)
                        curim = images[i].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        imstd = np.array([0.229, 0.224, 0.225])
                        immean = np.array([0.485, 0.456, 0.406])
                        curim = curim*imstd + immean
                        # print(curim.shape)
                        # print(curim[:, :, 0].mean())
                        # import ipdb
                        # ipdb.set_trace()
                        plt.imshow(curim[:, :, 0].astype(int), cmap = 'gray')


                        if notree:
                            plt.title([int(labels[i][0]), labels[i][1], labels[i][2]])
                        else:
                            plt.title(str(labels[i]))
                    plt.colorbar()
                    for j in range(6):
                        # second row
                        # chm
                        plt.subplot(2, 6, j+7)
                        curim = images[j].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        plt.imshow(curim[:, :, -1]*curim[:,:,0], cmap = 'gray')
                    # plt.show(block=False)

                elif not add_seg:
                    plt.figure(figsize = (20,8)) # size(width, height)
                    for j in range(6):
                        # second row
                        # chm
                        plt.subplot(2, 6, j+1)
                        curim = images[j].numpy()
                        curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end
                        plt.imshow(curim[:, :, -1], cmap = 'gray')

                        if notree:
                            plt.title([int(labels[j][0]), labels[j][1], labels[j][2]])
                        else:
                            plt.title(str(labels[j]))
                    plt.colorbar()
                    # plt.show(block=False)

    if twoinput:
        return images, input2
    else:
        return images


