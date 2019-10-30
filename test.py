#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.28 : 上午 10:08
@File Name: test.py
@Software: PyCharm
-----------------
"""
import numpy as np
import torch
from prepareDataset import ReIDDataset
from buildModel import AMOCNet
import itertools as it


def compute_cmc(dataset: ReIDDataset, cmc_test_inds, net: AMOCNet, sample_seq_length):
    print("Computing CMC metric")
    n_person = len(dataset)
    avgSame = 0
    avgDiff = 0
    avgSameCount = 0
    avgDiffCount = 0
    simMat = torch.zeros((n_person, n_person))
    cam = np.random.randint(0,2)
    rng = range(len(cmc_test_inds))
    if torch.cuda.is_available():
        net = net.cuda()
        simMat = simMat.cuda()

    with torch.no_grad():
        for shiftx in range(8):
            for doflip in range(2):
                shifty = shiftx
                for i, j in list(it.permutations(rng, 2)) + list(zip(rng, rng)):
                    indA, indB = cmc_test_inds[i], cmc_test_inds[j]
                    img_and_ofA = dataset.load_sequence_images(
                        *dataset.dataset[dataset.inds_set[indA]][cam] + (shiftx, shifty, doflip))
                    img_and_ofB = dataset.load_sequence_images(
                        *dataset.dataset[dataset.inds_set[indB]][cam] + (shiftx, shifty, doflip))
                    seqLenA = img_and_ofA.shape[0]
                    seqLenB = img_and_ofB.shape[0]
                    actual_sample_seq_length = min(seqLenA, seqLenB, sample_seq_length)
                    print(actual_sample_seq_length)
                    seq_length = actual_sample_seq_length
                    seqA = img_and_ofA[:seq_length - 1, :3, ...].copy()
                    seqB = img_and_ofB[:seq_length - 1, :3, ...].copy()
                    if len(seqA.shape) == 4:
                        seqA = np.expand_dims(seqA, 0)
                    if len(seqB.shape) == 4:
                        seqB = np.expand_dims(seqB, 0)

                    dist, _, _ = net(seqA, seqB).double()
                    assert len(dist.shape) == 3
                    simMat[i][j] += dist.data.double()
                    if i == j:
                        avgSame += dist.data.double()
                        avgSameCount += 1
                    else:
                        avgDiff + dist.data.double()
                        avgDiffCount += 1

        avgSame /= avgSameCount
        avgDiff /= avgDiffCount

        cmcInds = torch.DoubleTensor(n_person)
        cmc = torch.zeros(n_person).cuda()
        samplingOrder = torch.zeros((n_person,n_person)).cuda()
        for i in range(n_person):
            cmcInds[i] = i
            _, o = torch.sort(simMat[i, :])

            # find the element we want
            indx = (o == i).nonzero()
            # build the sampling order for the next epoch
            # we want to sample close images i.e. ones confused with this person
            samplingOrder[i][:o.shape[0] - 1] = list(filter(lambda x: x != i, o))
            for j in range(indx, n_person):
                cmc[j] += 1
        cmc = (cmc / n_person) * 100
        cmcString = ''
        for c in range(50):
            if c <= n_person:
                cmcString = cmcString + ' ' + torch.floor(cmc[c])
        print(cmcString)

    return cmc.data.cpu().numpy(), simMat.data.cpu().numpy(), samplingOrder.data.cpu().numpy(), avgSame, avgDiff
