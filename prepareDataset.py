#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.25 : 下午 8:38
@File Name: prepareDataset.py
@Software: PyCharm
-----------------
"""

import torch
import torch.utils.data.dataset as d
import os
import re
import cv2
import numpy as np
import dataset_utils as du
import logging as log


class ReIDDataset(d.Dataset):
    def __init__(self, t: int,
                 seq_root_rgb: str,
                 seq_root_of: str,
                 file_ext: str,
                 sample_seq_length: int,
                 disable_opticalflow: bool = False,
                 use_predefined: bool = True):
        super(ReIDDataset, self).__init__()

        assert t in [0, 1], ValueError
        self.dataset_name = du.DATASET_NAMES[t]
        self.seqRootRGB = seq_root_rgb
        self.seqRootOF = seq_root_of
        self.file_ext = file_ext
        self.sample_seq_length = sample_seq_length
        self.disable_opticalflow = disable_opticalflow
        self.person_dirs = self.get_person_dirs_list(self.seqRootRGB)
        self.dataset = self.load_dataset()

        log.info('loading predefined test/training split')
        self.train_inds, self.test_inds = du.partition_dataset(len(self.dataset), 0.98, t, use_predefined)
        self.inds_set = list(self.train_inds) + list(self.test_inds)

    def load_dataset(self) -> list:
        persons = []
        letter = ['a', 'b']
        for i, pdir in enumerate(self.person_dirs):
            persons += [[]]
            for cam in range(2):
                camera_dir_name = 'cam' + (str(cam + 1) if self.dataset_name == du.DATASET_NAMES[0] else letter[cam])
                seq_root = os.path.join(self.seqRootRGB, camera_dir_name, pdir)
                seq_of_root = os.path.join(self.seqRootOF, camera_dir_name, pdir)
                seq_imgs = self.get_sequence_image_files(seq_root)
                persons[i] += [(seq_root, seq_of_root, seq_imgs)]
            if self.dataset_name == du.DATASET_NAMES[1] and i == 200:
                break
        return persons

    def get_sequence_image_files(self, seq_root: str):
        seq_files = []
        for file in os.listdir(seq_root):
            if self.file_ext in file:
                seq_files.append(file)
        if not len(seq_files):
            return []

        return sorted(seq_files, key=lambda x: int(re.findall(r"([0-9]+)\.", x)[0]))

    def load_sequence_images(self, camera_dir: str, opticalflow_dir: str, files_list: list, *args) -> np.array:
        w, h = 64, 128
        assert len(files_list) > 1
        images = np.zeros((len(files_list), h, w, 5))
        for i, file in enumerate(files_list):
            filename = os.path.join(camera_dir, file)
            filename_of = os.path.join(opticalflow_dir, file)

            img = cv2.imread(filename)
            img = cv2.resize(img, (w, h))

            of = cv2.imread(filename_of)
            of = cv2.resize(of, (w, h))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            images[i, ..., :3] = img
            images[i, ..., 3:] = of[..., :2] * (1 - int(self.disable_opticalflow))
        images = du.data_augment(images, args)
        return images

    def get_person_dirs_list(self, data_root) -> list:
        first_camera_dirname = 'cam1' if self.dataset_name == du.DATASET_NAMES[0] else 'cam_a'
        person_dirs = []
        for file in os.listdir(os.path.join(data_root, first_camera_dirname)):
            if len(file) > 2:
                person_dirs.append(file)

        if not len(person_dirs):
            return []

        return sorted(person_dirs, key=lambda x: int(re.findall(r"([0-9]+)", x)[0]))

    def get_single(self, person: int, cam, start, sample_len):
        img_of = self.load_sequence_images(*self.dataset[person][cam])
        img = img_of[start: min(start + sample_len, img_of.shape[0]), :3, ...]
        of = img_of[start: min(start + sample_len, img_of.shape[0]), 3:, ...]
        return img, of

    def __len__(self):
        return len(self.inds_set)

    def __getitem__(self, ind: int):
        get_seq_pair_method = du.get_pos_sample if ind % 2 else du.get_neg_sample
        personA, personB, camA, camB, startA, startB, actual_sample_seqLen = get_seq_pair_method(
            self.dataset, self.inds_set[ind], self.sample_seq_length, self.inds_set)
        inputA, ofA = self.get_single(personA, camA, startA, actual_sample_seqLen)
        inputB, ofB = self.get_single(personB, camB, startB, actual_sample_seqLen)
        # print(F"input shapes: inputA: {inputA.shape}, inputB: {inputB.shape}, ind: {ind}")
        return inputA, inputB, np.array([1 if ind % 2 else -1]), personA, personB, ind, ofA, ofB
