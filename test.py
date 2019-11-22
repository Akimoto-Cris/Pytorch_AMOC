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

    shiftx_range = 6
    doflip_range = 2

    with torch.no_grad():
        with tqdm(total=shiftx_range * doflip_range * n_person ** 2) as t:
            t.set_description("Computing CMC metric")
            for shiftx in range(shiftx_range):
                for doflip in range(doflip_range):
                    shifty = shiftx
                    _img_and_of = [dataset.load_sequence_images(
                        *dataset.dataset[ind][0] + (shiftx, shifty, doflip)) for ind in cmc_test_inds]
                    actual_sample_seq_length = [min(io.shape[0], sample_seq_length) for io in _img_and_of]
                    img_and_of = [torch.from_numpy(expand_dim(io[:actual_sample_seq_length[i], :3, ...])).float().cuda()
                                  for i, io in enumerate(_img_and_of)]

                    for i, j in list(it.permutations(rng, 2)) + list(zip(rng, rng)):
                        seqA = img_and_of[i]
                        seqB = img_and_of[j]

                        dist, _, _ = net(seqA, seqB)
                        dist_v = dist.squeeze().data
                        simMat[i][j] += dist_v
                        if i == j:
                            avgSame += dist_v
                            avgSameCount += 1
                        else:
                            avgDiff += dist_v
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


def expand_dim(array: np.array) -> np.array:
    if len(array.shape) == 4:
        array = np.expand_dims(array, 0)
    return array
