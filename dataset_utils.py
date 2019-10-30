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
import cv2

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
        print('N train = %d' % len(datasetSplit["train_inds"]))
        print('N test  = %d' % len(datasetSplit["test_inds"]))
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


def get_pos_sample(dataset, person, sample_seqLen, *args):
    # choose the camera, ilids video only has two, but change this for other datasets
    camA = 0
    camB = 1

    nSeqA = len(dataset[person][camA][2])
    nSeqB = len(dataset[person][camB][2])

    actual_sample_seqLen = min(sample_seqLen, nSeqA, nSeqB)
    startA = np.random.randint(nSeqA - actual_sample_seqLen)
    startB = np.random.randint(nSeqB - actual_sample_seqLen)
    # print(nSeqA, nSeqB)

    return person, person, camA, camB, startA, startB, actual_sample_seqLen


def get_neg_sample(dataset, person_dumi, sample_seqLen, *args):
    assert len(args) > 0
    perm_all_persons = np.random.permutation(args[0])
    personA = perm_all_persons[0]
    personB = perm_all_persons[1]

    # choose the camera, ilids video only has two, but change this for other datasets
    camA = np.random.randint(low=0, high=2)
    camB = np.random.randint(low=0, high=2)

    nSeqA = len(dataset[personA][camA][2])
    nSeqB = len(dataset[personB][camB][2])

    actual_sample_seqLen = min(sample_seqLen, nSeqA, nSeqB)
    startA = np.random.randint(nSeqA - actual_sample_seqLen)
    startB = np.random.randint(nSeqB - actual_sample_seqLen)

    # print(nSeqA, nSeqB)
    return personA, personB, camA, camB, startA, startB, actual_sample_seqLen


def normalize(img: np.array) -> np.array:
    for c in range(img.shape[-1]):
        v = np.sqrt(np.var(img[..., c]))
        m = np.mean(img[..., c])
        img[..., c] -= m
        img[..., c] /= np.sqrt(v)
    return img


def data_augment(seq, *args, use_torch: bool = False):
    cropx, cropy, hflip = random_choices() if len(args) != 3 else args
    seqLen, seqDim1, seqDim2, seqChnls = seq.shape
    # print(seq.shape)
    daData = np.zeros(seq.shape) if not use_torch else torch.zeros(seq.shape)
    if use_torch and torch.cuda.is_available():
        daData = daData.cuda()
    for t in range(seqLen):
        thisFrame = seq[t, ...].copy()
        if hflip == 1:
            thisFrame = thisFrame[::-1, :, :]

        thisFrame = thisFrame[cropy: cropy + 128 - 8, cropx: cropx + 64 - 8, :]
        thisFrame = normalize(thisFrame)
        daData[t, ...] = cv2.resize(thisFrame, (seqDim2, seqDim1))
    daData = np.transpose(daData, [0, 3, 2, 1])
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

