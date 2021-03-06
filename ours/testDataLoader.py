#coding=gbk
from EPN3D_dataset import EPN3DDataset
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import importlib
import shutil

def main():
    '''DATA LOADING'''
    #log_string('Load dataset ...')
    DATA_PATH = '' # TO BE Modified

    TRAIN_DATASET = EPN3DDataset(root=DATA_PATH, class_choice=None, split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=1)
    '''
    for batch_id, data in enumerate(trainDataLoader,0):
        print(batch_id)
        points, target = data
        print(type(point),type(target))'''
    for batch_id, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        (points, target) = data
        points = points.data.numpy()
        print(type(points),type(target))
        print(points.shape,target.shape)
        print(target.data)
    '''
    print(type(trainDataLoader))
    print(len(trainDataLoader))
    print(trainDataLoader)'''

if __name__ == '__main__':
    main()

   
