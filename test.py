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
from tqdm import tqdm


def compute_cmc(dataset: ReIDDataset, cmc_test_inds, net: AMOCNet, sample_seq_length):
    n_person = len(cmc_test_inds)
    avgSame = 0
    avgDiff = 0
    avgSameCount = 0
    avgDiffCount = 0
    simMat = torch.zeros((n_person, n_person))
    rng = range(n_person)
    if torch.cuda.is_available():
        net = net.cuda()
        simMat = simMat.cuda()

    shiftx_range = 1
    doflip_range = 1

    with torch.no_grad():
        with tqdm(total=shiftx_range * doflip_range * n_person ** 2) as t:
            t.set_description("Computing CMC metric")
            for shiftx in range(shiftx_range):
                for doflip in range(doflip_range):
                    shifty = shiftx
                    for i, j in list(it.permutations(rng, 2)) + list(zip(rng, rng)):
                        indA, indB = cmc_test_inds[i], cmc_test_inds[j]
                        img_and_ofA = dataset.load_sequence_images(
                            *dataset.dataset[indA][0] + (shiftx, shifty, doflip))
                        img_and_ofB = dataset.load_sequence_images(
                            *dataset.dataset[indB][1] + (shiftx, shifty, doflip))
                        seqLenA = img_and_ofA.shape[0]
                        seqLenB = img_and_ofB.shape[0]
                        actual_sample_seq_lengthA = min(seqLenA, sample_seq_length)
                        actual_sample_seq_lengthB = min(seqLenB, sample_seq_length)
                        seqA = img_and_ofA[:actual_sample_seq_lengthA, :3, ...].copy()
                        seqB = img_and_ofB[:actual_sample_seq_lengthB, :3, ...].copy()
                        if len(seqA.shape) == 4:
                            seqA = np.expand_dims(seqA, 0)
                        if len(seqB.shape) == 4:
                            seqB = np.expand_dims(seqB, 0)

                        seqA = torch.Tensor(seqA).cuda()
                        seqB = torch.Tensor(seqB).cuda()
                        dist, outA, outB = net(seqA, seqB)
                        assert len(dist.shape) == 1, ValueError("Wrong output shape of distance:", dist.shape)
                        # print(f"gt: {cmc_test_inds[i]} - {cmc_test_inds[j]}, dist: {dist[0].data}")
                        dist = dist.squeeze()
                        simMat[i][j] += dist.data
                        if i == j:
                            avgSame += dist.data
                            avgSameCount += 1
                        else:
                            avgDiff += dist.data
                            avgDiffCount += 1
                        t.update(1)

        avgSame /= avgSameCount
        avgDiff /= avgDiffCount

        cmcInds = torch.DoubleTensor(n_person)
        cmc = torch.zeros(n_person).cuda()
        samplingOrder = torch.zeros((n_person, n_person)).cuda()
        for i in range(n_person):
            cmcInds[i] = i
            _, o = torch.sort(simMat[i, :])

            # find the element we want
            indx = (o == i).nonzero()
            # build the sampling order for the next epoch
            # we want to sample close images i.e. ones confused with this person
            samplingOrder[i][:o.shape[0] - 1] = torch.Tensor(list(filter(lambda x: x != i, o))).cuda()
            for j in range(indx, n_person):
                cmc[j] += 1
        cmc = (cmc / n_person) * 100

    return cmc.data.cpu().numpy(), simMat.data.cpu().numpy(), samplingOrder.data.cpu().numpy(), avgSame, avgDiff

