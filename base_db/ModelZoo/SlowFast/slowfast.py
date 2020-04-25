import os
import sys
import torch
import torch.nn as nn

## self define file
if __name__=='__main__':
    sys.path.append(os.path.abspath("../../"))
    from config import ActivityConfig as cfg
    from transform import *
else:
    from config import ActivityConfig as cfg
    from transform import *

MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class I3DBottleneck(nn.Module):
    ''' i3d bottleneck '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_temp_conv=0):
        super(I3DBottleneck, self).__init__()

        if use_temp_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1),
                                   padding=(1, 0, 0), bias=False)
        else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # self.non_local = non_local
        # if self.non_local == 1:
        #     #  self.NL = NONLocalBlock3D(in_channels=planes*4, sub_sample=False)
        #     self.nl_block = NONLocalBlock3D(in_channels=planes*4, sub_sample=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        # if self.non_local:
        #     #  out = self.NL(out)
        #     out = self.nl_block(out)

        return out

class SlowFastNet(nn.Module):
    ''' basic resnet structure '''
    def __init__(self, block, layers, num_classes):
        super(SlowFastNet, self).__init__()

        self.alpha = 8
        self.beta = 8
        self.num_classes = num_classes

        temp_conv_set_slow = [0, 0, 1, 1]
        temp_conv_set_fast = [1, 1, 1, 1]

        # slow pathway
        self.inplanes_slow = 64+64//self.beta*2

        self.conv1_slow = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1_slow = nn.BatchNorm3d(64)
        self.relu_slow = nn.ReLU(inplace=True)
        self.maxpool_slow = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)

        self.layer1_slow = self._make_layer_slow(block, 64, layers[0],
                                                 use_temp_conv=temp_conv_set_slow[0])

        #  non-local add pooling after res2
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer2_slow = self._make_layer_slow(block, 128, layers[1], stride=2,
                                                 use_temp_conv=temp_conv_set_slow[1])

        self.layer3_slow = self._make_layer_slow(block, 256, layers[2], stride=2,
                                                 use_temp_conv=temp_conv_set_slow[2])

        self.layer4_slow = self._make_layer_slow(block, 512, layers[3], stride=2,
                                                 use_temp_conv=temp_conv_set_slow[3])

        self.avgpool_slow = nn.AvgPool3d(kernel_size=(4, 7, 7), stride=1)

        # fast pathway
        self.inplanes_fast = 8

        self.conv1_fast = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1_fast = nn.BatchNorm3d(8)
        self.relu_fast = nn.ReLU(inplace=True)
        self.maxpool_fast = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)

        self.layer1_fast = self._make_layer_fast(block, 8, layers[0],
                                                 use_temp_conv=temp_conv_set_fast[0])

        #  non-local add pooling after res2
        # self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer2_fast = self._make_layer_fast(block, 16, layers[1], stride=2,
                                                 use_temp_conv=temp_conv_set_fast[1])

        self.layer3_fast = self._make_layer_fast(block, 32, layers[2], stride=2,
                                                 use_temp_conv=temp_conv_set_fast[2])

        self.layer4_fast = self._make_layer_fast(block, 64, layers[3], stride=2,
                                                 use_temp_conv=temp_conv_set_fast[3])

        self.avgpool_fast = nn.AvgPool3d(kernel_size=(32, 7, 7), stride=1)

        # lateral
        self.lateral_pool = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(self.alpha, 1 ,1),
                                      bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32, 32*2, kernel_size=(5, 1, 1), stride=(self.alpha, 1 ,1),
                                      bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64, 64*2, kernel_size=(5, 1, 1), stride=(self.alpha, 1 ,1),
                                      bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128, 128*2, kernel_size=(5, 1, 1), stride=(self.alpha, 1 ,1),
                                      bias=False, padding=(2, 0, 0))

        # self.bn2 = nn.BatchNorm3d(2048)
        self.dropout = nn.Dropout(p=cfg.TRAIN.DROPOUT)
        self.outplanes = 512*block.expansion + 64*block.expansion
        self.fullyconv = nn.Conv3d(self.outplanes, num_classes, kernel_size=1, stride=1)
        # self.softmax = nn.Softmax()

    def _make_layer_slow(self, block, planes, blocks, stride=1, use_temp_conv=None):
        downsample = None
        if stride != 1 or self.inplanes_slow != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes_slow, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_slow, planes, stride, downsample,
                            use_temp_conv=use_temp_conv))
        self.inplanes_slow = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_slow, planes,
                                use_temp_conv=use_temp_conv))

        self.inplanes_slow += self.inplanes_slow//self.beta*2
        return nn.Sequential(*layers)

    def _make_layer_fast(self, block, planes, blocks, stride=1, use_temp_conv=None):
        downsample = None
        if stride != 1 or self.inplanes_fast != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes_fast, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_fast, planes, stride, downsample,
                            use_temp_conv=use_temp_conv))
        self.inplanes_fast = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_fast, planes,
                                use_temp_conv=use_temp_conv))

        return nn.Sequential(*layers)

    def forward(self, input):
        fast, lateral = self.fast_path(input[:, :, :, :, :])
        slow = self.slow_path(input[:, :, ::self.alpha, :, :], lateral)

        x = torch.cat([slow, fast], dim=1)
        x = self.dropout(x)
        x = self.fullyconv(x).squeeze()

        return x

    def fast_path(self, x):
        lateral = []

        x = self.conv1_fast(x)
        x = self.bn1_fast(x)
        x = self.relu_fast(x)
        x = self.maxpool_fast(x)
        lateral.append(self.lateral_pool(x))

        x = self.layer1_fast(x)
        lateral.append(self.lateral_res2(x))

        x = self.layer2_fast(x)
        lateral.append(self.lateral_res3(x))

        x = self.layer3_fast(x)
        lateral.append(self.lateral_res4(x))

        x = self.layer4_fast(x)
        x = self.avgpool_fast(x)

        return x, lateral


    def slow_path(self, x, lateral):
        x = self.conv1_slow(x)
        x = self.bn1_slow(x)
        x = self.relu_slow(x)
        x = self.maxpool_slow(x)

        x = torch.cat([x, lateral[0]], dim=1)
        x = self.layer1_slow(x)

        x = torch.cat([x, lateral[1]], dim=1)
        x = self.layer2_slow(x)

        x = torch.cat([x, lateral[2]], dim=1)
        x = self.layer3_slow(x)

        x = torch.cat([x, lateral[3]], dim=1)
        x = self.layer4_slow(x)
        x = self.avgpool_slow(x)

        return x

def SlowFast50(num_classes):
    """Constructs a ResNet-50 model.
    """
    slow_fast = SlowFastNet(I3DBottleneck, [3, 4, 6, 3], num_classes)
    return slow_fast

class SlowFast(torch.nn.Module):
    ''' construct i3d model and optim policy '''
    def __init__(self, num_class,num_segments,backbone='resnet50',pretrain=True):
        super(SlowFast, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.backbone = backbone
        self.pretrain = pretrain
        # self.use_sync_bn = use_sync_bn
        self._prepare_base_model()
        ##TODO:collect model info
        cfg.IMG.CROP_SIZE = self.crop_size
        cfg.IMG.SCALE_SIZE = self.scale_size
        cfg.IMG.MEAN = self.input_mean
        cfg.IMG.STD = self.input_std

    def _prepare_base_model(self):
        if self.backbone == 'resnet50':
            if self.pretrain is not None:
                if self.pretrain in cfg.PRETRAIN_TYPE_LIST:
                    self.base_model = SlowFast50(self.num_class)
                else:
                    print("=> pretrain type %s don't support it" % (self.pretrain))
                    sys.exit(0)
            else:
                self.base_model =  SlowFast50(self.num_class)
            # self.base_model = convertBNtoSyncBN(base_model) if self.use_sync_bn else base_model
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            print("I3D backbone %s don't support it"%(self.backbone))
            sys.exit(0)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SlowFast, self).train(mode)
        #  count = 0
        #  for m in self.base_model.modules():
            #  if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                #  count += 1
                #  if count >= (2 if self._enable_pbn else 1):
                    #  m.eval()

                    #  # shutdown update in frozen mode
                    #  m.weight.requires_grad = False
                    #  m.bias.requires_grad = False

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
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult':0,
             'name': "normal_bias"},
            {'params': bns, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        input = input.view((-1, 3, self.num_segments) +input.size()[-2:])
        base_out = self.base_model(input)
        if len(base_out.shape) == 1:
            base_out = torch.unsqueeze(base_out, 0)
        # shape [N, CLASS_NUM]
        return base_out.squeeze(1)

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



if __name__ == '__main__':
    ## unit test
    cfg.MODEL_NAME = 'slowfast'
    cfg.BACKBONE = 'resnet50'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]

    x = torch.randn([1, 3, 32, 224, 224])
    net = SlowFast(cfg.DATASET.CLASS_NUM,
                   32,
                   backbone=cfg.BACKBONE,
                   pretrain= cfg.PRETRAIN_TYPE)
    out = net(x)
    print(out)
    ## uint test, measure FLOPs,Params
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)