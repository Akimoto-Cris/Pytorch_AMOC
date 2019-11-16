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

import os
from options import parser_args
from torch.utils.data.sampler import SubsetRandomSampler
from prepareDataset import ReIDDataset
import logging as log
from buildModel import *
from train import train_sequence, load_weight
from test import compute_cmc
from torchnet.logger import VisdomLogger, VisdomPlotLogger
import pickle


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
        seqRootRGB = os.path.join(dataset_root, 'i-LIDS-VID', 'sequences')
        seqRootOF = os.path.join(dataset_root, 'i-LIDS-VID-OF-HVP', 'sequences')
    else:
        seqRootRGB = os.path.join(dataset_root, 'PRID2011', 'multi_shot')
        seqRootOF = os.path.join(dataset_root, 'PRID2011-OF-HVP', 'multi_shot')

    log.info('loading Dataset - ', seqRootRGB, seqRootOF)
    reid_set = ReIDDataset(opt.dataset, seqRootRGB, seqRootOF, 'png', opt.sampleSeqLength, split_ratio=opt.data_split)
    log.info('Dataset loaded', len(reid_set))

    train_loader = torch.utils.data.DataLoader(reid_set, 1, num_workers=1,
                                               sampler=SubsetRandomSampler(list(range(len(reid_set)))))
    train_loader.dataset.inds_set = reid_set.train_inds
    test_loader = torch.utils.data.DataLoader(reid_set, 1, num_workers=1,
                                              sampler=SubsetRandomSampler(list(range(len(reid_set.test_inds)))))
    test_loader.dataset.inds_set = reid_set.test_inds

    # build the model
    fullModel = AMOCNet(len(reid_set.train_inds), opt)
    contrastive_criterion = ContrastiveLoss(opt.hingeMargin)
    class_criterion_A = nn.CrossEntropyLoss()
    class_criterion_B = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        fullModel = fullModel.cuda()
        contrastive_criterion = contrastive_criterion.cuda()
        class_criterion_A = class_criterion_A.cuda()
        class_criterion_B = class_criterion_B.cuda()

    # -- Training
    if opt.train:
        trained_model, trainer_log, val_history = train_sequence(fullModel, contrastive_criterion, class_criterion_A,
                                                                 class_criterion_B, train_loader, test_loader, opt)

    if opt.pretrained:
        trained_model = load_weight(fullModel, opt.pretrained, verbose=True)

    # -- Evaluation
    nTestImages = reid_set.test_inds[:30]  # [2 ** (n+1) for n in range(5)]

    cmc, simMat, _, avgSame, avgDiff = compute_cmc(reid_set, nTestImages, trained_model, 128)
    print(cmc)
    print(simMat)
    print(avgSame, avgDiff)
    sim_logger = VisdomLogger('heatmap', port=8097, opts={
        'title': 'simMat',
        'columnnames': list(range(len(simMat[0]))),
        'rownames': list(range(len(simMat)))
    })
    cmc_logger = VisdomPlotLogger("line", win="cmc_curve")
    for i, v in enumerate(cmc):
        cmc_logger.log(i, v, name="cmc_curve")
    sim_logger.log(simMat)

    log.info("Saving results...")
    with open("cmc.pkl", 'w') as f:
        pickle.dump(cmc, f)
    with open("simMat.pkl", 'w') as f:
        pickle.dump(simMat, f)
