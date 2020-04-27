'''
=> Description:
This file define I3D model based on ResNet. It contains class as follows:
* I3DBottleneck(class)
* I3DResNet(class)
* i3dinit(function)
* I3DR50(class)
=> Logical process:
I3DBottleneck--> I3DResNet--> I3DR50--> I3D
'''
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms

## self define file
if __name__=='__main__':
    sys.path.append(os.path.abspath("../../"))
    from config import ActivityConfig as cfg
    from transform import *
    from non_local import NONLocalBlock3D
else:
    from config import ActivityConfig as cfg
    from transform import *
    from .non_local import NONLocalBlock3D



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

    def __init__(self, inplanes, planes, stride=1, downsample=None, non_local=0, use_temp_conv=0):
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

        self.non_local = non_local
        if self.non_local == 1:
            #  self.NL = NONLocalBlock3D(in_channels=planes*4, sub_sample=False)
            self.nl_block = NONLocalBlock3D(in_channels=planes*4, sub_sample=False)

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

        if self.non_local:
            #  out = self.NL(out)
            out = self.nl_block(out)

        return out

class I3DResNet(nn.Module):
    ''' basic resnet structure '''
    def __init__(self, block, layers, num_classes, non_local_set, use_temp_conv_set):
        self.inplanes = 64
        self.num_classes = num_classes
        super(I3DResNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       non_local=non_local_set[0],
                                       use_temp_conv=use_temp_conv_set[0])

        #  non-local add pooling after res2
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       non_local=non_local_set[1],
                                       use_temp_conv=use_temp_conv_set[1])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       non_local=non_local_set[2],
                                       use_temp_conv=use_temp_conv_set[2])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       non_local=non_local_set[3],
                                       use_temp_conv=use_temp_conv_set[3])

        #  last_duration = math.ceil(cfg.TRAIN.FRAMES_IN_SNIPPET/2)
        #  last_size = math.ceil(cfg.TRAIN.IMAGE_WIDTH/32)

        self.avgpool = nn.AvgPool3d(kernel_size=(4, 7, 7), stride=1)

        self.dropout = nn.Dropout(p=cfg.TRAIN.DROPOUT)

        self.fullyconv = nn.Conv3d(512*block.expansion, num_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(-1)

    def _make_layer(self, block, planes, blocks, stride=1, non_local=None, use_temp_conv=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            non_local=non_local[0], use_temp_conv=use_temp_conv[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                non_local=non_local[i],
                                use_temp_conv=use_temp_conv[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # non-local add pooling after res2
        x = self.maxpool2(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #  x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fullyconv(x).squeeze()
        # x = self.softmax(x)

        if batch_size == 1:
            x = x.unsqueeze(0)

        return x

def i3dinit(model,backbone_pretrain_path=''):
    ''' inflated 3d model init '''
    ## TODO: load pretrain model
    _path = os.path.join(backbone_pretrain_path)
    if not os.path.exists(_path):
        print("=> %s don't exist!!!!"%(_path))
        sys.exit(0)
    pre_dict = torch.load(_path)
    own_state = model.state_dict()
    init_params = {}
    uninit_params = {}

    for opr_name, param in pre_dict.items():
        init_params[opr_name] = 1
        if 'fc' in opr_name:
            torch.nn.init.normal_(own_state[opr_name.replace('fc', 'fullyconv')], std=0.01)
            continue

        if isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        if param.dim() == own_state[opr_name].dim():
            own_state[opr_name].copy_(param)
        else:
            assert param.dim() == 4 and own_state[opr_name].dim() == 5, 'conv layer only'
            inflate = own_state[opr_name].shape[2]
            view_shape = param.shape[:2]+(1,)+param.shape[2:]
            own_state[opr_name].copy_(param.view(view_shape).repeat(1, 1, inflate, 1, 1)/inflate)

    for opr_name in own_state.keys():
        if opr_name not in init_params:
            uninit_params[opr_name] = 1

def I3DR50(num_classes,pretrain,backbone_pretrain_path=''):
    """Constructs a ResNet-50 model.
    """
    use_temp_convs_2 = [1, 1, 1]
    non_local_2 = [0, 0, 0]

    use_temp_convs_3 = [1, 0, 1, 0]
    non_local_3 = [0, 1, 0, 1]

    use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
    non_local_4 = [0, 1, 0, 1, 0, 1]

    use_temp_convs_5 = [0, 1, 0]
    non_local_5 = [0, 0, 0]

    non_local_set = [non_local_2, non_local_3, non_local_4, non_local_5]
    use_temp_conv_set = [use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]

    i3d_resnet50 = I3DResNet(I3DBottleneck, [3, 4, 6, 3], num_classes,
                             non_local_set=non_local_set, use_temp_conv_set=use_temp_conv_set)
    if pretrain:
        i3dinit(i3d_resnet50,backbone_pretrain_path=backbone_pretrain_path)

    return i3d_resnet50

class I3D(torch.nn.Module):
    def __init__(self, num_class,
                       num_segments,
                       backbone='resnet50',
                       pretrain='imagenet',
                       backbone_pretrain_path=''):
        super(I3D, self).__init__()
        self.num_class = num_class
        self.num_segments = num_segments
        self.backbone = backbone
        self.pretrain = pretrain
        self.backbone_pretrain_path = backbone_pretrain_path
        # self.use_sync_bn = use_sync_bn
        self._prepare_base_model()
        ## TODO:collect model info
        cfg.IMG.CROP_SIZE = self.crop_size
        cfg.IMG.SCALE_SIZE = self.scale_size
        cfg.IMG.MEAN = self.input_mean
        cfg.IMG.STD = self.input_std

    def _prepare_base_model(self):
        if self.backbone == 'resnet50':
            if self.pretrain is not None:
                if self.pretrain in cfg.PRETRAIN_TYPE_LIST:
                    self.base_model = I3DR50(self.num_class,
                                             True,
                                             backbone_pretrain_path=self.backbone_pretrain_path)
                else:
                    print("=> pretrain type %s don't support it" % (self.pretrain))
                    sys.exit(0)
            else:
                self.base_model = I3DR50(self.num_class, False)
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
        super(I3D, self).train(mode)
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
    if not os.path.exists(cfg.PRETRAIN_MODEL_ZOO):
        print("=> %s don't exist,will be created!!!" % (cfg.PRETRAIN_MODEL_ZOO))
        os.makedirs(cfg.PRETRAIN_MODEL_ZOO)

    cfg.MODEL_NAME = 'i3d'
    cfg.BACKBONE = 'resnet50'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]

    x = torch.randn([1,3,8,224,224])
    net = I3D(cfg.DATASET.CLASS_NUM,
              8,
              backbone = cfg.BACKBONE,
              pretrain = cfg.PRETRAIN_TYPE,
              backbone_pretrain_path=os.path.join(cfg.PRETRAIN_MODEL_ZOO,cfg.TRAIN.PRETRAIN_MODEL))
    out = net(x)
    print(out)
    ## uint test, measure FLOPs,Params
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)


