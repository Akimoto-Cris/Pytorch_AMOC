#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" 
@Author: Xu Kaixin
@License: Apache Licence 
@Time: 2019.10.26 : 下午 11:22
@File Name: buildModel.py
@Software: PyCharm
-----------------
"""
import torch
import torch.nn as nn

CONV_INIT_METHOD = nn.init.xavier_normal_
LINEAR_INIT_METHOD = nn.init.normal_


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout: float):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.actdrop = nn.Sequential(nn.Tanh(), nn.Dropout(dropout))
        self.N = nn.Linear(hidden_size, hidden_size, bias=False)
        self.M = nn.Linear(input_size, hidden_size, bias=False)

    def init_hidden(self):
        self.hidden = torch.zeros(1, self.hidden_size)
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = self.M(input_) + self.N(self.hidden)
        self.hidden = self.actdrop(output)
        return output


class AMOCNet(nn.Module):
    def __init__(self, n_persons_train, opt):
        super(AMOCNet, self).__init__()
        self.img_spat_net = SpatialNet(3, 16, opt.nConvFilters, stepSize=[2, 1])
        self.of_spat_net = SpatialNet(2, 16, opt.nConvFilters)
        self.fusion = FusionNet(5, opt.nConvFilters * 2, opt)
        self.motion_net = MotionNet([64, 64, 128, 128, 256, 256])
        self.rnn = MyRNN(input_size=opt.embeddingSize, hidden_size=opt.embeddingSize, dropout=opt.dropoutFrac)
        self.classifier = nn.Sequential(
            nn.Linear(opt.embeddingSize, n_persons_train),
            nn.LogSoftmax(dim=1)
        )
        self.distancer = nn.PairwiseDistance(2)

        self.img_spat_net = init_weights(self.img_spat_net)
        self.of_spat_net = init_weights(self.of_spat_net)
        self.fusion = init_weights(self.fusion)
        self.rnn = init_weights(self.rnn)
        self.classifier = init_weights(self.classifier)
        # print(self)
        if opt.motionnet_pretrained:
            self.motion_net.load_weight(opt.motionnet_pretrained)

    def forward_single(self, a):
        out_as = []
        self.rnn.init_hidden()
        for i in range(a.shape[1] - 1):
            img_a = self.img_spat_net(a[:, i, ...])
            consecutive_frame = torch.cat((a[:, i, ...], a[:, i + 1, ...]), 1)
            _, _, of = self.motion_net(consecutive_frame)
            pf_a = self.of_spat_net(of)
            fc_a = self.fusion(img_a, pf_a)
            out_as.append(self.rnn(fc_a))
        out_as = torch.cat(tuple(out_as), 0)
        temp_pool_a = torch.mean(out_as, 0)
        return torch.unsqueeze(temp_pool_a, 0), self.classifier(temp_pool_a.unsqueeze(0))

    def forward(self, a, b):
        temp_pool_a, output_a = self.forward_single(a)
        temp_pool_b, output_b = self.forward_single(b)
        distance = self.distancer(temp_pool_a, temp_pool_b)
        return distance, output_a, output_b


class SpatialNet(nn.Module):
    def __init__(self, input_dim, n_fltrs1, n_fltrs2, stepSize: list = None):
        super(SpatialNet, self).__init__()
        nFilters = [n_fltrs1, n_fltrs2]
        filtsize = [5, 5]
        poolsize = [2, 2]
        self.stepSize = [1, 1] if stepSize is None else stepSize
        self.input_dim = input_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, nFilters[0], filtsize[0], self.stepSize[0], filtsize[0] // 2),
            nn.Tanh(),
            nn.MaxPool2d(poolsize[0], 2),

            nn.Conv2d(nFilters[0], nFilters[1], filtsize[1], self.stepSize[1], filtsize[1] // 2),
            nn.Tanh(),
            nn.MaxPool2d(poolsize[1], 2),
        )

    def forward(self, input_):
        assert input_.shape[1] == self.input_dim
        return self.cnn(input_)


class FusionNet(nn.Module):
    poolsize = 2
    stepSize = 2

    def __init__(self, filter_size, n_filter, opt, fusion_method: str = "cat"):
        super(FusionNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_filter, opt.embeddingSize, filter_size, 1, filter_size // 2),
            nn.Tanh(),
        )
        nFullyConnected = opt.embeddingSize * 8 * 16

        self.fc = nn.Sequential(
            nn.Dropout(opt.dropoutFrac),
            nn.Linear(nFullyConnected, opt.embeddingSize)
        )
        self.fusion_method = fusion_method

    def _fuse(self, a, b):
        if self.fusion_method == "cat":
            return torch.cat((a, b), 1)
        elif self.fusion_method == "max":
            raise NotImplementedError(f"Only concatenate method ('cat') is implemented currently.")
        else:
            raise NotImplementedError(f"Only concatenate method ('cat') is implemented currently.")

    def forward(self, a, b):
        x = self._fuse(a, b)
        x = self.cnn(x)
        return self.fc(torch.reshape(x, [1, -1]))


class MotionNet(nn.Module):
    def __init__(self, n_filters: list):
        super(MotionNet, self).__init__()
        filtsize = [7, 3, 5, 3, 5, 3]
        stepSize = 2

        # constructing feature pyramid
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, n_filters[0], filtsize[0], stepSize, filtsize[0] // 2), nn.Tanh())
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_filters[0], n_filters[1], filtsize[1], 1, filtsize[1] // 2), nn.Tanh())
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters[1], n_filters[2], filtsize[2], stepSize, filtsize[2] // 2), nn.Tanh())
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(n_filters[2], n_filters[3], filtsize[3], 1, filtsize[3] // 2), nn.Tanh())
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_filters[3], n_filters[4], filtsize[4], stepSize, filtsize[4] // 2), nn.Tanh())
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(n_filters[4], n_filters[5], filtsize[5], 1, filtsize[5] // 2), nn.Tanh())

        # decoder for optical flow vector field
        self.pred1 = nn.Sequential(
            nn.Conv2d(n_filters[5], 2, 3, 1, 1), nn.Tanh())
        self.pred_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 4, stepSize, 1), nn.Tanh())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(n_filters[5], n_filters[3], 4, stepSize, 1), nn.Tanh())

        self.pred2 = nn.Sequential(
            nn.Conv2d(n_filters[3] * 2 + 2, 2, 3, 1, 1), nn.Tanh())
        self.pred_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, 4, stepSize, 1), nn.Tanh())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters[2] * 2 + 2, n_filters[0], 4, stepSize, 1), nn.Tanh())

        # final output optical flow vector field
        self.pred3 = nn.Sequential(
            nn.Conv2d(n_filters[0] * 2 + 2, 2, 3, 1, 1), nn.Tanh())

    def load_weight(self, path, verbose: bool = True):
        if verbose:
            print("loading motion net weights from", path)
        foreign_state_dict = torch.load(path)
        target_state_dict = self.state_dict()
        cnt = 0
        for key, v in foreign_state_dict.items():
            matched_key = [k for k in target_state_dict.keys() if k in key]
            if matched_key:
                target_state_dict[matched_key[0]] = foreign_state_dict[key]
                cnt += 0
        if verbose:
            print(f"loaded {cnt} layers from {path}")

    def forward(self, input_: torch.Tensor):
        assert input_.shape[1] == 6
        x = input_
        x_1 = self.conv1_1(self.conv1(x))
        x_2 = self.conv2_1(self.conv2(x_1))
        x_3 = self.conv3_1(self.conv3(x_2))

        pred_1 = self.pred1(x_3)
        pred_deconv_1 = self.pred_deconv1(pred_1)

        deconv_1 = self.deconv1(x_3)
        deconv_1 = torch.cat((deconv_1, pred_deconv_1, x_2), 1)

        pred_2 = self.pred2(deconv_1)
        pred_deconv_2 = self.pred_deconv2(pred_2)

        deconv_2 = self.deconv2(deconv_1)
        deconv_2 = torch.cat((deconv_2, pred_deconv_2, x_1), 1)

        pred_3 = self.pred3(deconv_2)
        return pred_1, pred_2, pred_3       # let intermediate byproducts available for pretraining


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            LINEAR_INIT_METHOD(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            CONV_INIT_METHOD(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


