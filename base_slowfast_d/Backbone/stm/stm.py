# -*- coding: utf-8 -*-
# @Time    : 19-2-25 下午8:49
# @Author  : Boyuan Jiang
# @File    : STM.py


import torch.nn as nn
import torch
import pdb
from .CMM import CMM
import os
import sys
if __name__ == '__main__':
    sys.path.append("../../")
    from config import ActivityConfig as cfg
else:
    from config import ActivityConfig as cfg

__all__ = ['STM', 'stm50', 'stm101', 'stm152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, T=8):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.T = T
        self.shift = nn.Conv1d(planes, planes, 3, padding=1, groups=planes, bias=False)
        weight = torch.zeros(planes, 1, 3)
        weight[:planes//4,0,0]=1.0
        weight[planes//4:planes//4+planes//2,0,1]=1.0
        weight[-planes//4:,0,2]=1.0
        self.shift.weight = nn.Parameter(weight)
        self.mb = CMM(planes, T)
        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        identity = x

        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out_ = self.mb(out)
        if self.stride==2:
            out_ = self.maxpool(out_)
        # [NT,C,H,W]
        shape=out.shape
        NT=shape[0]
        C=shape[1]
        H=shape[2]
        W=shape[3]
        # [N,T,C,H,W]
        out = out.view(-1,self.T,C,H,W)
        # [N,H,W,C,T]
        out = out.permute(0,3,4,2,1)
        # [NHW,C,T]
        out = out.contiguous().view(-1, C, self.T)
        out = self.shift(out)
        # [N,H,W,C,T]
        out = out.view(-1, H, W, C, self.T)
        # [N,T,C,H,W]
        out = out.permute(0,4,3,1,2)
        # [NT,C,H,W]
        out = out.contiguous().view(NT, C, H, W)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out+out_

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class STM(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, dropout=0, T=8, with_fc=True, consensus='avg', img_feature_dim=256):
        super(STM, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], T=T)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, T=T)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, T=T)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, T=T)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if with_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.p = dropout
        self.with_fc = with_fc
        self.consensus = consensus
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d): #or isinstance(m, SyncBatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, T=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, T=T))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, T=T))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.p!=0:
            x = self.dropout(x)
        if self.with_fc:
            x = self.fc(x)

        return x

    def load_parameters(self, init, ckpt, prefix='', strict=True):
        if init == 'scratch':
            print("-----from scratch-----")
        elif init == "imagenet":
            unexpected_keys = ['fc.weight', 'fc.bias']
            pretrained_state_dict = torch.load(ckpt, map_location='cpu')
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k not in unexpected_keys}
            model_dict = self.state_dict()
            model_dict.update(pretrained_state_dict)
            self.load_state_dict(model_dict)
            if self.with_fc:
                nn.init.normal_(self.fc.weight, 0, 0.001)
                nn.init.constant_(self.fc.bias, 0)
            del pretrained_state_dict
            print("-----from imagenet-----")
        elif init == 'kinetics':
            unexpected_keys = 'fc'
            pretrained_state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if unexpected_keys not in k}
            pretrained_state_dict = {k[18:]: v for k, v in pretrained_state_dict.items()}
            model_dict = self.state_dict()
            model_dict.update(pretrained_state_dict)
            self.load_state_dict(model_dict)
            if self.with_fc:
                nn.init.normal_(self.fc.weight, 0, 0.001)
                nn.init.constant_(self.fc.bias, 0)
            del pretrained_state_dict
            print("-----from kinetics-----")
        else:
            raise NotImplementedError

def _STM(arch, block, layers, pretrained, **kwargs):
    model = STM(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        ## TODO: load pretrain model
        if not os.path.exists(cfg.PRETRAIN_MODEL_ZOO):
            print("=> %s don't exist,will be created!!!" % (cfg.PRETRAIN_MODEL_ZOO))
            os.makedirs(cfg.PRETRAIN_MODEL_ZOO)
        _path = os.path.join(cfg.PRETRAIN_MODEL_ZOO, cfg.TRAIN.PRETRAIN_MODEL)
        if not os.path.exists(_path):
            print("=> %s don't exist"%(_path))
            sys.exit(0)
        model.load_parameters(cfg.PRETRAIN_TYPE,_path)
        # pretrain_dict = torch.load(_path)
        # model.load_state_dict(pretrain_dict)
    return model


def stm50(pretrained=False, **kwargs):
    """Constructs a stm-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _STM('stm50',Bottleneck, [3, 4, 6, 3], pretrained,**kwargs)
    # if pretrained is not '':
    #     print("loading kinetics pretrained model from: ", pretrained)
    #     model.load_state_dict(torch.load(pretrained), strict=False)
    # else:
    #     model.load_state_dict(model_zoo.load_url(model_urls['TSM50']))
    return model


def stm101(pretrained=False, **kwargs):
    """Constructs a stm-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _STM('stm101',Bottleneck, [3, 4, 23, 3],pretrained, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['TSM101']))
    return model


def stm152(pretrained=False, **kwargs):
    """Constructs a stm-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _STM('stm152',Bottleneck, [3, 8, 36, 3],pretrained ,**kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['TSM152']))
    return model


if __name__ == '__main__':
    ## unit test
    ## TODO: config: MODEL_NAME:tsn  BACKBONE:resnet50
    cfg.MODEL_NAME = 'tsn'
    cfg.BACKBONE = 'stm50'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]
    x = torch.randn([8,3,224,224])
    net = stm50(pretrained=True)
    out = net(x)
    print(out)
