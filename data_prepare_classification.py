#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:39:15 2022

@author: sizhuo
"""

f = '/media/RS_storage/Users/Xiaoye/greenhouse/validation/labels.csv'

import pandas as pd
d = pd.read_csv(f) 

d.head()

d['useCase'] = 0

import random
for i in range(len(d)):
    if random.random()<0.10:
        
        d.loc[i]['useCase'] = 'test'
        
    else:
        d.loc[i]['useCase'] = 'train'
   
    
for i, row in d.iterrows():
    if random.random()<0.08:
        
        d.at[i,'useCase'] = 'test'
    else:
        d.at[i,'useCase'] = 'train'
        

for i, row in d.iterrows():
    # print(d.loc[i]['path'])
    raw = d.loc[i]['path']
    new = raw.replace(r'C:\Users\cwm937\validation','/media/RS_storage/Users/Xiaoye/greenhouse/validation')
    
    new = new.replace(r'\s', '/s')
    print(new)
    
    
    
    d.at[i,'path'] = new



d.useCase.value_counts()

d.to_csv(f)
600/3272