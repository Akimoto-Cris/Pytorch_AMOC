#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.26 : 上午 11:33
@File Name: dataset_utils.py
@Software: PyCharm
-----------------
"""
import torch
import pickle as pkl
import numpy as np
import math
import os

DATASET_NAMES = ["ILIDS-VID", "PRID2011"]


def save_pickle(filename: str, obj):
    assert filename.split(".")[-1] == "pkl", ValueError
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(filename: str):
    assert filename.split(".")[-1] == "pkl", ValueError
    with open(filename, 'rb') as f:
        file = pkl.load(f)
    return file


def partition_dataset(n_total_persons: int, test_train_split: float, dataset_type: int, load_predefined: bool = True):
    assert 0 <= test_train_split < 1, ValueError
    partition_file_name = f'trainedNets\\datasplit_{DATASET_NAMES[dataset_type]}.pkl'
    if os.path.exists(partition_file_name) and load_predefined:
        datasetSplit = load_pickle(partition_file_name)
        return datasetSplit["train_inds"], datasetSplit["test_inds"]

    split_point = int(np.floor(n_total_persons * test_train_split))
    inds = np.random.permutation(n_total_persons)

    # save the inds to a pickle file
    save_pickle('rnnInds.pkl', inds)

    train_inds = inds[:split_point]
    test_inds = inds[split_point:]

    print('N train = %d' % len(train_inds))
    print('N test  = %d' % len(test_inds))

    # save the split to a file for later use
    datasetSplit = {
        "train_inds": train_inds,
        "test_inds": test_inds,
    }

    save_pickle(partition_file_name, datasetSplit)
    return train_inds, test_inds


# the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
# choose a pair of sequences from the same person
def get_pos_sample(dataset, person, sample_seqLen):
    # choose the camera, ilids video only has two, but change this for other datasets
    camA = 0
    camB = 1

    nSeqA = len(dataset[person][camA][2])
    nSeqB = len(dataset[person][camB][2])

    actual_sample_seqLen = min(sample_seqLen, nSeqA, nSeqB)
    startA = np.random.randint(max(nSeqA - actual_sample_seqLen, 0))
    startB = np.random.randint(max(nSeqB - actual_sample_seqLen, 0))
    print(sample_seqLen, nSeqA, nSeqB, startA, startB)

    return startA, startB, actual_sample_seqLen


# the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
# choose a pair of sequences from different people
def get_neg_sample(dataset, train_inds, sample_seqLen):
    perm_all_persons = np.random.permutation(len(train_inds))
    personA = perm_all_persons[0]
    personB = perm_all_persons[1]

    # choose the camera, ilids video only has two, but change this for other datasets
    camA = math.floor(np.random.rand(1)[0] * 2)
    camB = math.floor(np.random.rand(1)[0] * 2)

    nSeqA = len(dataset[personA][camA][2])
    nSeqB = len(dataset[personB][camB][2])

    actual_sample_seqLen = min(sample_seqLen, nSeqA, nSeqB)
    startA = np.random.randint(max(nSeqA - actual_sample_seqLen, 0))
    startB = np.random.randint(max(nSeqB - actual_sample_seqLen, 0))

    print(sample_seqLen, nSeqA, nSeqB, startA, startB)
    return personA, personB, camA, camB, startA, startB, actual_sample_seqLen


def normalize(img: np.array) -> np.array:
    for c in range(img.shape[-1]):
        v = np.sqrt(np.var(img[..., c]))
        m = np.mean(img[..., c])
        img[..., c] -= m
        np.true_divide(img[..., c], np.sqrt(v))
    return img


def data_augment(seq, *args, use_torch: bool = False):
    cropx, cropy, hflip = random_choices() if len(args) != 3 else args
    seqLen, seqChnls, seqDim1, seqDim2 = seq.shape
    data_size = (seqLen, seqChnls, seqDim1 - 8, seqDim2 - 8)
    daData = np.zeros(data_size) if not use_torch else torch.zeros(data_size)
    if use_torch and torch.cuda.is_available():
        daData = daData.cuda()
    for t in range(seqLen):
        thisFrame = seq[t, ...].squeeze().copy()
        if hflip == 1:
            thisFrame = thisFrame[:, ::-1, :]

        thisFrame = thisFrame[:, cropx: cropx + 64 - 8, cropy: cropy + 128 - 8]
        thisFrame = normalize(thisFrame)
        daData[t, ...] = thisFrame
    return daData


def random_choices():
    crpx = math.floor(np.random.rand(2)[0] * 8)
    crpy = math.floor(np.random.rand(2)[0] * 8)
    flip = math.floor(np.random.rand(2)[0] * 2)
    return crpx, crpy, flip


def to_categorical(ind, tot_length: int):
    if "__iter__" not in dir(ind):
        assert ind < tot_length
        ind = [ind]
    else:
        assert max(ind) < tot_length
    one_hot = [[int(i == n) for n in range(tot_length)] for i in ind]
    return np.array(one_hot).T

