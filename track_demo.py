#!/usr/bin/python3  
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.11.06 : 下午 8:24
@File Name: track_demo.py
@Software: PyCharm
-----------------
"""

import cvlib as cv
from dataset_utils import normalize, denormalize
from prepareDataset import ReIDDataset
import cv2
import os
import numpy as np
from buildModel import AMOCNet
import torch


class Detection:
    def __init__(self):
        self.bbox: list
        self.id = None
        self.sequence = []
        self.color = None
        self.sequence_tensor = None
        self.max_lenth: int = 128
        self.seq_len: int = 16

    def update(self, frame):
        if len(self.sequence) == self.max_lenth - 1:
            self.sequence = self.sequence[1:]   # FIFO
        self.sequence += [frame]
        self._update_tensor()
        self._write_disk()

    def _update_tensor(self):
        """
        Get a sequence of images containing single person.
        if the sequence length in storage is shorter than 2,
        the FIRST frame will be padded as a prefix to match the requested length.
        """
        if self.sequence_tensor is None:    # initialize
            pre_stack = [self.sequence[0]] * 2
            stacked = np.stack(pre_stack, axis=0)
            self.sequence_tensor = torch.Tensor(stacked)
            if torch.cuda.is_available():
                self.sequence_tensor = self.sequence_tensor.cuda()
        else:
            # discard the oldest frame, append new frame to the end
            new = torch.Tensor(self.sequence[-1]).unsqueeze(0)
            if torch.cuda.is_available():
                new = new.cuda()
            if self.sequence_tensor.shape[0] < self.seq_len:
                self.sequence_tensor = torch.cat((self.sequence_tensor, new), 0)
            else:
                self.sequence_tensor = torch.cat((self.sequence_tensor[1:, ...], new), 0)

    def _write_disk(self):
        dir_name = f"runtime_detect"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        person_dir_name = f"person_{self.id}"
        if not os.path.exists(os.path.join(dir_name, person_dir_name)):
            os.mkdir(os.path.join(dir_name, person_dir_name))
        path_name = f"{len(self.sequence) - 1}.png"
        cv2.imwrite(os.path.join(dir_name, person_dir_name, path_name),
                    cv2.cvtColor(denormalize(self.sequence[-1]), cv2.COLOR_YUV2BGR))


class Identifier:
    def __init__(self, model: AMOCNet, testset: ReIDDataset, seq_len: int = 128):
        self.model = model
        self.testset = testset
        self.gallery_inds = self.testset.test_inds
        self.seq_len = seq_len

    def request(self, prob: torch.Tensor):
        if len(prob.shape) == 4:
            prob = prob.unsqueeze(0)

        # matching
        min_dist = np.inf
        fetched = 0
        for id in self.gallery_inds:
            seq, _ = self.testset.get_single(id, 1, np.random.randint(8), self.seq_len)

            gall = torch.Tensor(seq).unsqueeze(0)
            if torch.cuda.is_available():
                gall = gall.cuda()
            dist, _, _ = self.model(prob, gall)
            dist = float(dist.squeeze().data.cpu())
            if min_dist > dist:
                min_dist = dist
                fetched = id
        return fetched, min_dist


def draw_bbox(image, bbox, class_no, color):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 10)
    cv2.putText(image, "class: " + str(class_no), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)


def frame_from_list(source, img_size):
    for file in os.listdir(source):
        filename = os.path.join(source, file)
        img = cv2.imread(filename)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = normalize(img)
        yield img


def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)


def check_siamese_model(source,
                        model: AMOCNet,
                        dataset: ReIDDataset,
                        image_size,
                        model_extractor='yolov3-tiny',
                        sample_sequence_len: int = 128,
                        memory_size=1,
                        start=0,
                        confidence=0.1,
                        margin=1):
    counter = 0
    id_counter = 0

    identifier = Identifier(model, dataset, sample_sequence_len)

    if os.path.isdir(source):
        video = frame_from_list(source, image_size)
    elif os.path.isfile(source):
        video = cv2.VideoCapture(source)
        if not video.isOpened():
            print("Could not open video")
            exit()
        status, frame = video.read()
        if not status:
            exit()
    else:
        raise FileNotFoundError("Video file does not exist, exiting")

    while True:
        counter += 1
        if os.path.isfile(source):
            status, frame = video.read()
            if not status:
                exit()
        else:
            try:
                frame = video.next()
            except StopIteration:
                break

        # skip number of frames
        if counter < start:
            continue

        bboxes, labels, confs = cv.detect_common_objects(frame, confidence=confidence, model=model_extractor)
        if len(bboxes):
            person_confs = [confs[i] if label == "person" else 0 for i, label in enumerate(labels)]

            detection = Detection()
            detection.bbox = bboxes[argmax(person_confs)]
            if detection.bbox[1] == detection.bbox[3] or detection.bbox[2] == detection.bbox[0]:
                continue
            detection.color = (255, 0, 255)
            detection.update(cv2.resize(frame[detection.bbox[1]: detection.bbox[3],
                                              detection.bbox[0]: detection.bbox[2]],
                                        image_size, interpolation=cv2.INTER_CUBIC))
            retrieved, dist = identifier.request(detection.sequence_tensor)
            if dist < margin:
                detection.id = retrieved
            else:
                detection.id = id_counter
                id_counter += 1
                identifier.gallery += [detection.sequence_tensor]