#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.25 : 下午 8:24
@File Name: train_motionnet.py
@Software: PyCharm
-----------------
"""

import torch
import os
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from prepareDataset import ReIDDataset
import logging as log
from buildModel import *
from train import train_motion_net
from test import compute_cmc
from videoReid import parser_args

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


def isnan(z):
    return z != z


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    opt = parser_args()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # change these paths to point to the place where you store i-lids or prid datasets
    dataset_root = "D:\\datasets"

    if opt.dataset == 0:
        seqRootRGB = os.path.join(dataset_root, 'i-LIDS-VID\\sequences')
        seqRootOF = os.path.join(dataset_root, 'i-LIDS-VID-OF-HVP\\sequences')
    else:
        seqRootRGB = os.path.join(dataset_root, 'PRID2011\\multi_shot')
        seqRootOF = os.path.join(dataset_root, 'PRID2011-OF-HVP\\multi_shot')

    log.info('loading Dataset - ',seqRootRGB,seqRootOF)
    reid_set = ReIDDataset(opt.dataset, seqRootRGB, seqRootOF, 'png', opt.sampleSeqLength)
    log.info('Dataset loaded', len(reid_set))

    train_sampler = SubsetRandomSampler(reid_set.train_inds)
    test_sampler = SubsetRandomSampler(reid_set.test_inds)
    train_loader = torch.utils.data.DataLoader(reid_set, 1, num_workers=1, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(reid_set, 1, num_workers=1, sampler=test_sampler)

    # build the model

    fullModel = AMOCNet(len(train_loader.dataset), opt)
    criterion = nn.L1Loss()
    if torch.cuda.is_available():
        fullModel = fullModel.cuda()
        criterion = criterion.cuda()

    # -- Training
    trained_model, trainer_log, val_history = train_motion_net(fullModel, criterion, train_loader, test_loader, opt)
