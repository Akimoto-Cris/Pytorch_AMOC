#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.25 : 下午 8:24
@File Name: videoReid.py
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
from train import train_sequence
from test import compute_cmc

log.DEBUG

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpochs', type=int, default=500,  help='number of training epochs')
    parser.add_argument('--dataset', type=int, default=0, help='0 - ilids, 1 - prid')
    parser.add_argument('--sampleSeqLength', type=int, default=16, help='length of sequence to train network')
    # parser.add_argument('--gradClip', type=int, default=5, help='magnitude of clip on the RNN gradient')
    parser.add_argument('--saveFileName', type=str, default='amoc', help='name to save dataset file')
    parser.add_argument('--usePredefinedSplit', dest="usePredefinedSplit", action="store_true",
                        help='Use predefined test/training split loaded from a file')
    parser.add_argument('--dropoutFrac', type=float, default=0.3, help='fraction of dropout to use between layers')
    parser.add_argument('--dropoutFracRNN', type=float, default=0.3, help='fraction of dropout to use between RNN layers')
    parser.add_argument('--samplingEpochs', type=int, default=100,
                        help='how often to compute the CMC curve - dont compute too much - its slow!')
    parser.add_argument('--disableOpticalFlow', action="store_true", default=False, dest="disableOpticalFlow",
                        help='use optical flow features or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--learningRate', type=float, default=1e-4)
    parser.add_argument('--nConvFilters', type=int, default=32)
    parser.add_argument('--embeddingSize', type=int, default=128)
    parser.add_argument('--hingeMargin', type=int, default=2)
    parser.add_argument('--lr_decay', type=int, default=10000, help="Decay by 0.1 every `lr_decay` ITERATION")
    parser.add_argument('--checkpoint_path', type=str, default='trainedNets')
    parser.add_argument('--resume_epoch', type=int, default=0)
    options = parser.parse_args()
    return options


def isnan(z):
    return z != z


def main():
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
    contrastive_criterion = nn.HingeEmbeddingLoss(opt.hingeMargin)
    class_criterion_A = nn.BCELoss()
    class_criterion_B = nn.BCELoss()
    if torch.cuda.is_available():
        fullModel = fullModel.cuda()
        contrastive_criterion = contrastive_criterion.cuda()
        class_criterion_A = class_criterion_A.cuda()
        class_criterion_B = class_criterion_B.cuda()

    # -- Training
    trained_model, trainer_log, val_history = train_sequence(fullModel, contrastive_criterion, class_criterion_A,
                                                             class_criterion_B, train_loader, test_loader, opt)


    # -- Evaluation
    nTestImages = [2 ** (n+1) for n in range(8)]

    for n in nTestImages:
        log.info('test multiple images ', n)
        compute_cmc(reid_set, reid_set.test_inds, trained_model, n)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
