# -*- coding: utf-8 -*-
# @Time    : 19-2-25 下午12:00
# @Author  : Boyuan Jiang
# @File    : CMM.py

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CMM(nn.Module):
    def __init__(self, inp, T=7):
        super(CMM, self).__init__()
        self.T = T
        self.inp = inp
        self.reduce_time = 16
        self.stage_conv = conv1x1(inp, inp//self.reduce_time, 1)
        self.stage_bn = nn.BatchNorm2d(inp//self.reduce_time)
        self.shift = nn.Conv2d(inp//self.reduce_time, inp//self.reduce_time, 3, 1, 1, groups=inp//self.reduce_time, bias=False)
        self.conv1x1 = nn.Conv2d(inp//self.reduce_time, inp, 1, 1, bias=False)

    def forward(self, input):
        # build shift conv layer

        # input [N*T,C,H,W]
        reduced = self.stage_bn(self.stage_conv(input))
        shift1 = self.shift(reduced)
        # reshape [N*T,C,H,W]-->[N,T,C,H,W]
        inp_shape = reduced.shape
        NT = inp_shape[0]
        C = inp_shape[1]
        H = inp_shape[2]
        W = inp_shape[3]
        shift1 = shift1.view(-1, self.T, C, H, W)
        # GET T+\DELTA T shape [N,T-1,C,H,W]
        dshift1 = shift1[:, 1:, :, :, :]
        # GET R
        R1 = dshift1 - reduced.view(-1, self.T, C, H, W)[:,:-1]
        R = R1
        # pdb.set_trace()
        R = F.pad(R, (0,0,0,0,0,0,0,1,0,0))
        R = R.view(-1, C, H, W)
        R = self.conv1x1(R)

        return R