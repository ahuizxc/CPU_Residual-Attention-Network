#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 23:27:55 2018

@author: liushenghui
"""
import pickle
with open('cifar10/data_batch_1','rb') as f:
    dataset = pickle.load(f, encoding='bytes')
data = dataset['data']
label = dataset['label']
