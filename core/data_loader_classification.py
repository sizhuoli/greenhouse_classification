#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:23:49 2022

@author: sizhuo
"""

import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import rasterio
import glob
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
# from skimage.transform import resize
from scipy.ndimage import zoom
import cv2
import matplotlib.pyplot as plt
from random import sample
from scipy import ndimage
import ipdb
# load one or multiple input image sources, concatenate them (np array), convert to tensor image using ToTensor() and then feed
# into torch vision transformer (takes tensor image of shape (B, C, H, W))
def NDVI(r, n):
    """
    NDVI function.
    Inputs:
        * r - red array (np.array)
        * n - nir array (np.array)
    Output:
        * ndvi - ndvi array (np.array)
    """

    np.seterr(divide='ignore', invalid='ignore') # Ignore the divided by zero or Nan appears
    n = np.float32(n)
    r = np.float32(r)
    ndvi = (n-r)/(n+r) # The NDVI formula
    ndvi = np.float32(ndvi) # Convert datatype to float32 for memory saving.

    return (ndvi)


def EXGI(g, r, b):
    exgi = 2 * g -(r + b)
    exgi = np.float32(exgi)
    return exgi

def rgb2gray(rgb):

    r, g, b = rgb[0, :,:], rgb[1, :,:], rgb[2, :,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def warpAffine(src, M, dsize, from_bounding_box_only=False):
    """
    Applies cv2 warpAffine, marking transparency if bounding box only
    The last of the 4 channels is merely a marker. It does not specify opacity in the usual way.
    """
    return cv2.warpAffine(src, M, dsize)


class ImageFolder(data.Dataset):
    def __init__(self, root, config, mode='train', split_list = []):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.mode = mode
        self.config = config

        print('normalize input------')
        print('imageNet norm: ', self.config.imageNetnorm)
        print('meanstd norm: ', self.config.meanstdnorm)


        if self.mode == 'train':
            # trian and test all in csv file, with specific use case (train, test or na)
            self.GT_dd = pd.read_csv(self.config.classification_label_fn)
            extract_fields = []
            extract_fields.extend(['path', 'label', 'useCase'])
            extract_fields = list(set(extract_fields))

            self.GT_df = self.GT_dd[extract_fields]


        if self.mode == 'test':
            # trian and test all in csv file, with specific use case (train, test or na)
            self.GT_dd = pd.read_csv(self.config.classification_label_fn)

            # normal case
            extract_fields = []
            extract_fields.extend(['path', 'label', 'useCase'])
            extract_fields = list(set(extract_fields))

            self.GT_df = self.GT_dd[extract_fields]

            self.image_paths = glob.glob(f'{self.config.test_path}/**/*tif', recursive=True)
            test_list = self.GT_df.loc[self.GT_df['useCase']=='test', 'path'].to_list()

            self.image_paths = [f for f in self.image_paths if f in test_list]

            # remove samples in the testing set and also train set
            self.image_paths = [f for f in self.image_paths if f not in split_list]


        elif self.mode == 'pred':
            self.image_paths = glob.glob(f'{self.config.input_image_dir}/*tif', recursive=True)
            print('len of images', len(self.image_paths))
        else:
            self.image_paths = glob.glob(f'{self.config.train_path}/**/*tif', recursive=True)
            self.image_paths = [f for f in self.image_paths if f in split_list]
            print('len of images', len(self.image_paths))


        if self.mode == 'pred':
            self.image_paths = glob.glob(f'{self.config.train_path}/*.tif', recursive=True)

        if self.mode != 'test':
            self.augmentation_prob = self.config.augmentation_prob

        elif self.mode == 'test':
            self.augmentation_prob = 0

        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        filename = self.image_paths[index]
        image_path = filename

        if self.config.use_color:
            image = rasterio.open(image_path).read() # channel first
            image = np.stack((image[2, :, :], image[1, :, :], image[0, :, :]))
            if self.config.input_rescale:
                image = image
                image = image.astype('float32')


            # ipdb.set_trace()
            if self.config.add_ndvi:
                ndvi = NDVI(image[0, :, :], image[-1, :, :])


            if self.config.onlyRGB:
                image = image[:3, :, :]
                if self.config.rgb2gray:

                    if self.config.add_exgi:
                        # only when gary, exgi and chm bands_
                        exgi = EXGI(image[1, :, :], image[0, :, :], image[2, :, :])


                    image = rgb2gray(image)
                    if self.config.expandGray:
                        image = np.array([image]*2).astype(np.float32)
                    else:
                        image = np.array([image]*1).astype(np.float32)

                        if self.config.add_exgi:
                            exgi = np.array([exgi]*1).astype(np.float32)
                            image = np.concatenate((image, exgi), axis=0)
                        if self.config.add_ndvi:
                            ndvi = np.array([ndvi]*1).astype(np.float32)
                            image = np.concatenate((image, ndvi), axis=0)

        if self.config.mode != 'pred':
            # trian and test all in csv file, with specific use case (train, test or na)
            self.GT_dd = pd.read_csv(self.config.classification_label_fn)

            # normal case
            extract_fields = []
            extract_fields.extend(['path', 'label', 'useCase'])
            extract_fields = list(set(extract_fields))

            self.GT_df = self.GT_dd[extract_fields]

            GT = self.GT_df.loc[self.GT_df['path']==filename, 'label'].tolist()
            GT = GT[0]

            if GT == 'f':
                gt = 0
            elif GT == 'a':
                gt = 1
            # ipdb.set_trace()
            gt = torch.tensor(gt)

            # to fed into totensor, need channel last
            image = np.transpose(image, axes=(1,2,0)) # Channel at the end

            p_transform = random.random()
            # crop to circle first and then rotate, to maintain information
            # if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:
            #     rot_deg = random.sample([0, 90, 180, 270, 360], 1)
            #     image = ndimage.rotate(image, rot_deg, reshape=False)

            Transform = []
            Transform.append(T.ToTensor())

            if (self.mode == 'train') and p_transform <= self.config.augmentation_prob:
                # Transform.append(T.RandomRotation(180, fill=(0,)))
                Transform.append(T.RandomHorizontalFlip())
                Transform.append(T.RandomVerticalFlip())
                Transform = T.Compose(Transform)
                image = Transform(image)
                Transform =[]

            Transform.append(T.Resize((self.config.inputlength, self.config.inputlength)))

            if self.config.imageNetnorm:
                Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            Transform = T.Compose(Transform)
            # print('max', image.max())
            image = Transform(image)


            return image, gt
        else:
            # to fed into totensor, need channel last
            image = np.transpose(image, axes=(1,2,0)) # Channel at the end

            p_transform = random.random()
            Transform = []
            Transform.append(T.ToTensor())
            Transform.append(T.Resize((self.config.inputlength, self.config.inputlength)))

            if self.config.imageNetnorm:
                Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

            Transform = T.Compose(Transform)
            # print('max', image.max())
            image = Transform(image)


            return image

    def __len__(self):
        return len(self.image_paths)



def get_loader(image_path, config, mode, split_list):
    dataset = ImageFolder(root = image_path, config = config, mode = mode, split_list = split_list)

    # import ipdb
    # ipdb.set_trace()
    if mode == 'train' or mode == 'validation':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      )
    elif mode == 'pred':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=False,
                                      )
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      )
        # print(len(data_loader))
    return data_loader
