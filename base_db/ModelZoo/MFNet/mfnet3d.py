"""
Author: Yunpeng Chen
"""
import logging
import os
from collections import OrderedDict
import torch.nn as nn
import sys
import torch
import torchvision.transforms as torch_transforms
if __name__ == '__main__':
    sys.path.append(os.path.abspath("../../"))
    _path = os.path.abspath("../../")
    import initializer
    from transform import *
    from config import ActivityConfig as cfg
else:
    from . import initializer
    from transform import *
    from config import ActivityConfig as cfg


class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,
                               stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1,1,1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt,pt = (3,1) if use_3d else (1,0)
        # prepare input
        self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0))
        self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0))
        # main part
        self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)

    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class MFNET_3D(nn.Module):

    def __init__(self, num_classes=None,seg_num=None, pretrained=False, **kwargs):
        super(MFNET_3D, self).__init__()
        self.num_segments = seg_num
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        cfg.IMG.CROP_SIZE = self.crop_size
        cfg.IMG.SCALE_SIZE = self.scale_size
        cfg.IMG.MEAN = self.input_mean
        cfg.IMG.STD = self.input_std
        self.backbone = 'mfnet2d'

        groups = 16
        k_sec  = {  2: 3, \
                    3: 4, \
                    4: 6, \
                    5: 3  }

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv1_num_out if i==1 else conv2_num_out,
                                        num_mid=num_mid,
                                        num_out=conv2_num_out,
                                        stride=(2,1,1) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[2]+1)
                    ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv2_num_out if i==1 else conv3_num_out,
                                        num_mid=num_mid,
                                        num_out=conv3_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[3]+1)
                    ]))

        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv3_num_out if i==1 else conv4_num_out,
                                        num_mid=num_mid,
                                        num_out=conv4_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[4]+1)
                    ]))

        # conv5 - x7 (x8)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv4_num_out if i==1 else conv5_num_out,
                                        num_mid=num_mid,
                                        num_out=conv5_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[5]+1)
                    ]))

        # final
        self.tail = nn.Sequential(OrderedDict([
                    ('bn', nn.BatchNorm3d(conv5_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))

        self.globalpool = nn.Sequential(OrderedDict([
                        ('avg', nn.AvgPool3d(kernel_size=(8,7,7),  stride=(1,1,1))),
                        # ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
                        ]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)


        #############
        # Initialization
        initializer.xavier(net=self)

        if pretrained:
            import torch
            load_method='inflation' # 'random', 'inflation'
            ## TODO: load pretrain model
            if not os.path.exists(cfg.PRETRAIN_MODEL_ZOO):
                print("=> %s don't exist,will be created!!!"%(cfg.PRETRAIN_MODEL_ZOO))
                os.makedirs(cfg.PRETRAIN_MODEL_ZOO)
            _path = os.path.join(cfg.PRETRAIN_MODEL_ZOO,cfg.TRAIN.PRETRAIN_MODEL)
            if not os.path.exists(_path):
                print("=> %s don't exist!"%(_path))
                sys.exit(0)
            state_dict_2d = torch.load(_path)
            initializer.init_3d_from_2d_dict(net=self, state_dict=state_dict_2d, method=load_method)
        else:
            print("Network:: graph initialized, use random inilization!")

    def forward(self, x):
        x = x.view((-1, 3, self.num_segments) + x.size()[-2:])
        assert x.shape[2] == 16

        h = self.conv1(x)   # x224 -> x112
        h = self.maxpool(h) # x112 ->  x56

        h = self.conv2(h)   #  x56 ->  x56
        h = self.conv3(h)   #  x56 ->  x28
        h = self.conv4(h)   #  x28 ->  x14
        h = self.conv5(h)   #  x14 ->   x7

        h = self.tail(h)
        h = self.globalpool(h)

        h = h.view(h.shape[0], -1)
        h = self.classifier(h)

        return h

    def get_optim_policies(self):
        '''
        normal action:      weight --> conv + fc weight
                            bias   --> conv + fc bias
        bns:                all bn3.

        '''
        normal_weight = []
        normal_bias = []
        bns = []

        for mod in self.modules():

            if isinstance(mod, (torch.nn.Conv3d, torch.nn.Conv2d)):
                param = list(mod.parameters())
                normal_weight.append(param[0])
                if len(param) == 2:
                    normal_bias.append(param[1])

            elif isinstance(mod, torch.nn.Linear):
                param = list(mod.parameters())
                normal_weight.append(param[0])
                if len(param) == 2:
                    normal_bias.append(param[1])

            elif isinstance(mod, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d)):
                bns.extend(list(mod.parameters()))

            elif not mod._modules:
                if list(mod.parameters()):
                    raise ValueError("New atomic module type: {}. \
                                         Need to give it a learning policy".format(type(mod)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_feat"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bns, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def train_transform(self):
        return torch_transforms.Compose([
            GroupMultiScaleCrop(self.crop_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip(is_flow=False),
            Stack(roll=(self.backbone == 'BNInception')),
            ToTorchFormatTensor(div=(self.backbone != 'BNInception')),
            GroupNormalize(self.input_mean, self.input_std)])

    def val_transform(self):
        return torch_transforms.Compose([
            GroupScale(int(self.scale_size)),
            GroupCenterCrop(self.crop_size),
            Stack(roll=(self.backbone == 'BNInception')),
            ToTorchFormatTensor(div=(self.backbone != 'BNInception')),
            GroupNormalize(self.input_mean, self.input_std)])

if __name__ == "__main__":
    ## unit test
    cfg.MODEL_NAME = 'mfnet3d'
    cfg.BACKBONE = 'mfnet2d'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]

    x = torch.randn([1, 3, 16, 224, 224])
    net = MFNET_3D(num_classes=51,seg_num=16,pretrained=True)
    output = net(x)
    print(output)
    ## uint test, measure FLOPs,Params
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)