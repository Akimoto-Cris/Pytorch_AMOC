#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.11.06 : 下午 8:24
@File Name: options.py
@Software: PyCharm
-----------------
"""
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpochs', type=int, default=1000,  help='number of training epochs')
    parser.add_argument('--dataset', type=int, default=0, help='0 - ilids, 1 - prid')
    parser.add_argument('--sampleSeqLength', type=int, default=16, help='length of sequence to train network')
    parser.add_argument('--gradClip', type=int, default=5, help='magnitude of clip on the RNN gradient')
    parser.add_argument('--saveFileName', type=str, default='amoc', help='name to save dataset file')
    parser.add_argument('--usePredefinedSplit', dest="usePredefinedSplit", action="store_true", default=False,
                        help='Use predefined test/training split loaded from a file')
    parser.add_argument('--dropoutFrac', type=float, default=0.3, help='fraction of dropout to use between layers')
    parser.add_argument('--dropoutFracRNN', type=float, default=0.3, help='fraction of dropout to use between RNN layers')
    parser.add_argument('--samplingEpochs', type=int, default=2000,
                        help='how often to compute the CMC curve - dont compute too much - its slow!')
    parser.add_argument('--disableOpticalFlow', action="store_true", default=False, dest="disableOpticalFlow",
                        help='use optical flow features or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--learningRate', '-l', type=float, default=1e-3)
    parser.add_argument('--nConvFilters', type=int, default=32)
    parser.add_argument('--embeddingSize', type=int, default=128)
    parser.add_argument('--hingeMargin', type=int, default=2)
    parser.add_argument('--lr_decay', type=int, default=10000, help="Decay by 0.1 every `lr_decay` ITERATION")
    parser.add_argument('--checkpoint_path', type=str, default='trainedNets')
    parser.add_argument('--motionnet_pretrained', '-mp', type=str)
    parser.add_argument('--pretrained', '-p', type=str)
    parser.add_argument('--data_split', '-ds', type=float, default=0.5)
    parser.add_argument('--train', dest="train", action="store_true")
    parser.add_argument('--source', dest="source", default=0, help="Camera_id or directory/to/frame_files or path/to/video")
    parser.add_argument('--tracker', dest="tracker", default='KCF', help="one of KCF and GOTURN")
    options = parser.parse_args()
    return options
