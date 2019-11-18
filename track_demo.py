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
import torch.nn as nn
from options import parser_args
from train import load_weight
import time


def preprocess(img: np.array) -> np.array:
    assert img.shape[-1] == 3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img.astype(np.float)
    img = normalize(img)
    img = np.transpose(img, [2, 1, 0])
    return img


class Detection:
    """
    Temporal-Spatial Tensor along detected trajectory of person
    """
    def __init__(self):
        self.sequence = []
        self.color = None
        self.sequence_tensor = None
        self.max_lenth: int = 4
        self.seq_len: int = 4
        self.id = None

    def update(self, frame):
        if len(self.sequence) == self.max_lenth - 1:
            self.sequence = self.sequence[1:]   # FIFO
        self.sequence += [preprocess(frame)]
        self._update_tensor()
        # self._write_disk()

    def _update_tensor(self):
        """
        Get a sequence of images containing single person.
        if the sequence length in storage is shorter than 2,
        the FIRST frame will be padded as a prefix to match the requested length.
        """
        if self.sequence_tensor is None:    # initialize
            pre_stack = [self.sequence[0]] * 2
            stacked = np.stack(pre_stack, axis=0)
            self.sequence_tensor = torch.from_numpy(stacked)
            if torch.cuda.is_available():
                self.sequence_tensor = self.sequence_tensor.cuda()
        else:
            # discard the oldest frame, append new frame to the end
            new = torch.from_numpy(self.sequence[-1]).float()
            new = new.unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                new = new.cuda()
            if self.sequence_tensor.shape[1] < self.seq_len:
                self.sequence_tensor = torch.cat((self.sequence_tensor, new), 1)
            else:
                self.sequence_tensor = torch.cat((self.sequence_tensor[:, 1:, ...], new), 1)
        if len(self.sequence_tensor.shape) < 5:
            self.sequence_tensor = self.sequence_tensor.unsqueeze(0).float()

    def _write_disk(self):
        assert self.id is not None, "Please assign id before calling update method."
        dir_name = f"runtime_detect"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.person_dir_name = os.path.join(dir_name, f"person_{self.id}")
        if not os.path.exists(self.person_dir_name):
            os.mkdir(self.person_dir_name)
        path_name = f"{len(self.sequence) - 1}.png"
        cv2.imwrite(os.path.join(self.person_dir_name, path_name),
                    cv2.cvtColor(denormalize(self.sequence[-1]), cv2.COLOR_YUV2BGR))

    def __add__(self, other):
        self.sequence += other.sequence
        self.sequence_tensor = torch.cat((self.sequence_tensor, other.sequence_tensor), 0)
        return self


class Identifier:
    def __init__(self, model: AMOCNet, seq_len: int = 128, testset: ReIDDataset = None, capacity: int = 120):
        self.model = model
        self.seq_len = seq_len
        self.capacity = capacity
        self.gallery_encodes = {}
        if testset:
            self.build_gallery(testset, testset.test_inds)
        self.distancer = nn.PairwiseDistance(2)

    def build_gallery(self, testset, gallery_inds):
        for _id in gallery_inds:
            seq, _ = testset.get_single(_id, 1, np.random.randint(8), self.seq_len)
            gall = torch.Tensor(seq).unsqueeze(0)
            if torch.cuda.is_available():
                gall = gall.cuda()
            if len(self) < self.capacity:
                self.gallery_encodes[_id] = gall

    def update_gallery(self, detect: Detection):
        """ (detection: Detection) -> id: int"""
        _id = len(self)
        if _id < self.capacity:
            begin = time.clock()
            encode, _ = self.model.forward_single(detect.sequence_tensor)
            print(time.clock() - begin)
            self.gallery_encodes[_id] = encode
        return _id

    def __len__(self):
        return len(self.gallery_encodes)

    def request(self, prob_encode: torch.Tensor):
        if len(prob_encode.shape) == 1:
            prob_encode = prob_encode.unsqueeze(0)
        if torch.cuda.is_available():
            prob_encode = prob_encode.cuda()

        # matching
        min_dist = np.inf
        fetched = 0
        for i, gall_encode in self.gallery_encodes.items():
            dist = self.distancer(gall_encode, prob_encode)
            dist = dist.item()
            if min_dist > dist:
                min_dist = dist
                fetched = i
        return fetched, min_dist


def draw_bbox(image, bbox, display_class, color):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
    cv2.putText(image, display_class, (bbox[0] + 2, bbox[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


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


def add_tracker(tracker_type):
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    else:
        raise NotImplementedError
    return tracker


def box_process(bbox, frame_size):
    return [
        int(max(bbox[0], 0)),
        int(max(bbox[1], 0)),
        int(min(bbox[2], frame_size[1])),
        int(min(bbox[3], frame_size[0]))
    ]


def main(source,
         model: AMOCNet,
         image_size,
         model_extractor='yolov3-tiny',
         sample_sequence_len: int = 128,
         tracker_type: str = 'KCF',
         dataset: ReIDDataset = None,
         confidence=0.1,
         margin=1, interval: int = 2):
    old_detection = {}
    old_tracker = {}
    identifier = Identifier(model, sample_sequence_len, testset=dataset)

    if isinstance(source, str) and os.path.isdir(source):
        video = frame_from_list(source, image_size)
    elif os.path.isfile(source) or isinstance(source, int):
        video = cv2.VideoCapture(source)
        if not video.isOpened():
            print("Could not open video")
            exit()
        status, frame = video.read()
        if not status:
            exit()
    else:
        raise FileNotFoundError("Video file does not exist, exiting")

    tracker_success = False
    counter = 0
    while True:
        counter += 1
        if isinstance(source, int) or os.path.isfile(source):
            status, frame = video.read()
            if not status:
                exit()
        else:
            try:
                frame = video.next()
            except StopIteration:
                break

        display_class = "person"

        # detect human, for now only support single human reid
        if not tracker_success:
            bboxes, labels, confs = cv.detect_common_objects(frame, confidence=confidence, model=model_extractor)
            if len(bboxes):
                detected = True
                person_confs = [confs[i] if label == "person" else 0 for i, label in enumerate(labels)]
                person_box = bboxes[argmax(person_confs)]
                tracker = add_tracker(tracker_type)
                tracker_success = tracker.init(frame, tuple(person_box))

                detection = Detection()
                detection.color = tuple([np.random.randint(3) * 255 // 2,
                                         np.random.randint(3) * 255 // 2,
                                         np.random.randint(3) * 255 // 2])
                detection.update(cv2.resize(frame[person_box[1]: person_box[3], person_box[0]: person_box[2]],
                                            image_size, interpolation=cv2.INTER_CUBIC))
                if len(identifier):
                    prob_encode, _ = model.forward_single(detection.sequence_tensor)
                    _id, dist = identifier.request(prob_encode)
                    old_detection[_id] = detection
                if not len(identifier) or dist >= margin:
                    _id = identifier.update_gallery(detection)
                    if _id in old_detection.keys():
                        old_detection[_id].update(detection.sequence[-1])
                    else:
                        old_detection[_id] = detection

                old_tracker[_id] = tracker
                draw_bbox(frame, person_box, display_class, detection.color)
            continue
        else:
            atleast_one = False
            for _id, (tracker, detect) in enumerate(zip(old_tracker.values(), old_detection.values())):
                # print(len(detect.sequence))
                # TODO: NMS
                detected, person_box = tracker.update(frame)
                person_box = box_process(person_box, frame.shape)
                atleast_one = atleast_one or detected
                if detected and 0 <= person_box[1] < person_box[3] and 0 <= person_box[0] < person_box[2]:
                    detect.update(cv2.resize(frame[person_box[1]: person_box[3], person_box[0]: person_box[2]],
                                             image_size, interpolation=cv2.INTER_CUBIC))
                    draw_bbox(frame, person_box, display_class + str(_id), detect.color)
                    break
            tracker_success = atleast_one

        if not tracker_success:
            continue

        # identify the trajectory from the gallery
        if not counter % interval:
            for _id, detect in old_detection.items():
                encode, _ = model.forward_single(detect.sequence_tensor)
                retrieved, dist = identifier.request(encode)
                if retrieved != _id and dist < margin:
                    old_detection[retrieved] = old_detection.pop(_id)
                    old_tracker[retrieved] = old_tracker.pop(_id)

        cv2.imshow("Person Video Re-Id", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parser_args()
    model = AMOCNet(150, opt)
    if opt.pretrained:
        trained_model = load_weight(model, opt.pretrained, verbose=True)
    if torch.cuda.is_available():
        fullModel = model.cuda()
    main(opt.source, model, (64, 128), model_extractor="yolov3")
