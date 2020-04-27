import yaml
import os
import sys
import torch
from torch import nn

## self define file
if __name__== '__main__':
    sys.path.append(os.path.abspath("../../"))
    _path =  os.path.abspath("../../")
    from config import ActivityConfig as cfg
    cfg.WORKSPACE_PATH = _path
    from layer_factory import get_basic_layer, parse_expr
else:
    from config import ActivityConfig as cfg
    from .layer_factory import get_basic_layer, parse_expr

class BNInception(nn.Module):
    def __init__(self, model_path='./bn_inception.yaml', num_classes=101,
                       weight_url='https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth',
                       pretrained = True
                 ):
        super(BNInception, self).__init__()

        model_path = os.path.join(cfg.WORKSPACE_PATH,"Backbone/bninception/bn_inception.yaml")
        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        if pretrained:
            # print("ok")
            new_state_dict = {}
            if not os.path.exists(cfg.PRETRAIN_MODEL_ZOO):
                print("=> %s don't exist,will be created!!!"%(cfg.PRETRAIN_MODEL_ZOO))
                os.makedirs(cfg.PRETRAIN_MODEL_ZOO)
            _path = os.path.join(cfg.PRETRAIN_MODEL_ZOO,cfg.TRAIN.PRETRAIN_MODEL)
            ## TODO: load pretrain model
            if not os.path.exists(_path):
                print("=> %s don't exist"%(_path))
                sys.exit(0)
            pretrain_dict = torch.load(_path)
            for k, v in pretrain_dict.items():
                if (k.split(".")[0] == 'last_linear'):
                    new_state_dict['fc.' + k.split(".")[1]] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=False)

        # self.load_state_dict(torch.utils.model_zoo.load_url(weight_url,model_dir="/data/model_zoo"))

    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        return data_dict[self._op_list[-1][2]]


class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)

if __name__ == "__main__":
    ## TODO: config: MODEL_NAME:tsn BACKBONE:BNInception
    cfg.MODEL_NAME = 'tsn'
    cfg.BACKBONE = 'BNInception'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]
    x = torch.randn([1,3,224,224])
    net = BNInception(pretrained=True)
    out = net(x)
    print(out)
