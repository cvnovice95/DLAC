import sys
import torch
import time
from torch.utils.data import DataLoader
from multiprocessing.managers import BaseManager
import torchvision.transforms as torch_transforms
## self-define file
from data_datasets import DPFlowDataset,TSNDataSet,HumanactionDataset,DPFlowUntrimmedDataset
from data_loader import DataLoaderX,SuperDataLoader,DPFlowDataLoaderDP,PipeDataLoaderDP
from config import ActivityConfig as cfg
from utils import GenerateDPFlowAddr,AverageMeter
from transform import *



def train_provider_remote(transform=None):
    if not (cfg.DATASET.LOAD_METHOD in cfg.DATASET.LOAD_METHOD_LIST):
        print("=> Load Method %s don't exist!!!,Please checking it."%(cfg.DATASET.LOAD_METHOD))
        sys.exit(0)
    if cfg.DATASET.LOAD_METHOD == 'DPFlowDataset':
        BaseManager.register('DPFlowDataset', DPFlowDataset)
        manager = BaseManager()
        manager.start()
        inst = manager.DPFlowDataset(dataset_name=cfg.REDIS_DATASET,
                                    mode='training',  # validation
                                    transformer=transform,
                                    sample_method=cfg.REDIS_TRAIN_SAMPLE_METHOD,
                                    seg_num_ext=cfg.REDIS_TRAIN_SEG_NUM)
        return inst
    elif cfg.DATASET.LOAD_METHOD == 'TSNDataSet':
        BaseManager.register('TSNDataSet', TSNDataSet)
        manager = BaseManager()
        manager.start()
        inst = manager.TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                                dataset_type='video',
                                meta_file_name=cfg.DATASET.TRAIN_META_PATH,
                                sample_method=cfg.DATASET.TRAIN_SAMPLE_METHOD,
                                seg_num_ext=cfg.TRAIN.SEG_NUM,
                                mode='train',
                                transform=transform,
                                modality=cfg.TRAIN.MODALITY,
                                img_format=cfg.DATASET.IMG_FORMART)
        return inst
    else:
        print("=> Invaild Load Method!!![{}].Support Load Method has:{}".format(cfg.DATASET.LOAD_METHOD,
                                                                                cfg.DATASET.LOAD_METHOD_LIST))
        sys.exit(0)
## TODO:train_base
def train_base(transform=None):
    if not (cfg.DATASET.LOAD_METHOD in cfg.DATASET.LOAD_METHOD_LIST):
        print("=> Load Method %s don't exist!!!,Please checking it." % (cfg.DATASET.LOAD_METHOD))
        sys.exit(0)

    if cfg.DATASET.LOAD_METHOD == 'DPFlowDataset':
        dataset = DPFlowDataset(dataset_name=cfg.REDIS_DATASET,
                                mode='training',  # validation
                                transformer=transform,
                                sample_method=cfg.REDIS_TRAIN_SAMPLE_METHOD,
                                seg_num_ext=cfg.REDIS_TRAIN_SEG_NUM)
        return dataset

    elif cfg.DATASET.LOAD_METHOD == 'TSNDataSet':
        dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                             dataset_type='video',
                             meta_file_name=cfg.DATASET.TRAIN_META_PATH,
                             sample_method=cfg.DATASET.TRAIN_SAMPLE_METHOD,
                             seg_num_ext=cfg.TRAIN.SEG_NUM,
                             mode='train',
                             transform=transform,
                             modality=cfg.TRAIN.MODALITY,
                             img_format=cfg.DATASET.IMG_FORMART)
        return dataset
    elif cfg.DATASET.LOAD_METHOD == 'HumanactionDataset':
        dataset = HumanactionDataset(pkl_root_path=cfg.HUMANACTION_PKL_ROOT_PATH,
                                    pkl_name=cfg.HUMANACTION_TRAIN_PKL_NAME,
                                    seg_num_ext=cfg.TRAIN.SEG_NUM ,
                                    transformer=transform,
                                    sample_method=None,
                                    mode='training')
        return dataset
    else:
        print("=> Invaild Load Method!!![{}].Support Load Method has:{}".format(cfg.DATASET.LOAD_METHOD,
                                                                                cfg.DATASET.LOAD_METHOD_LIST))
        sys.exit(0)
## TODO:val_base
def val_base(transform=None):
    if not (cfg.DATASET.LOAD_METHOD in cfg.DATASET.LOAD_METHOD_LIST):
        print("=> Load Method %s don't exist!!!,Please checking it." % (cfg.DATASET.LOAD_METHOD))
        sys.exit(0)

    if cfg.DATASET.LOAD_METHOD == 'DPFlowDataset':
        dataset = DPFlowDataset(dataset_name=cfg.REDIS_DATASET,
                                mode='validation',  # validation
                                transformer=transform,
                                sample_method=cfg.REDIS_VAL_SAMPLE_METHOD,
                                seg_num_ext=cfg.REDIS_VAL_SEG_NUM)
        return dataset

    elif cfg.DATASET.LOAD_METHOD == 'TSNDataSet':
        dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                             dataset_type='video',
                             meta_file_name=cfg.DATASET.VAL_META_PATH,
                             sample_method=cfg.DATASET.VAL_SAMPLE_METHOD,
                             seg_num_ext=cfg.VAL.SEG_NUM,
                             mode='val',
                             transform=transform,
                             modality=cfg.VAL.MODALITY,
                             img_format=cfg.DATASET.IMG_FORMART)
        return dataset

    elif cfg.DATASET.LOAD_METHOD == 'HumanactionDataset':
        dataset = HumanactionDataset(pkl_root_path=cfg.HUMANACTION_PKL_ROOT_PATH,
                                     pkl_name=cfg.HUMANACTION_VAL_PKL_NAME,
                                     seg_num_ext=cfg.VAL.SEG_NUM,
                                     transformer=transform,
                                     sample_method=None,
                                     mode='validation')
        return dataset

    elif cfg.DATASET.LOAD_METHOD == 'DPFlowUntrimmedDataset':
        dataset = DPFlowUntrimmedDataset(dataset_name=cfg.REDIS_DATASET,
                                mode='validation',  # validation
                                transformer=transform,
                                sample_method=cfg.REDIS_VAL_SAMPLE_METHOD,
                                seg_num_ext=cfg.REDIS_VAL_SEG_NUM)
        return dataset
    else:
        print("=> Invaild Load Method!!![{}].Support Load Method has:{}".format(cfg.DATASET.LOAD_METHOD,
                                                                                cfg.DATASET.LOAD_METHOD_LIST))
        sys.exit(0)

def train_loader_local(transform=None):
    dataset = train_base(transform=transform)
    if cfg.USE_DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoaderX(dataset,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             num_workers=8,
                             pin_memory=False,
                             sampler=train_sampler)
        if cfg.SUPER_LOADER:
            print("=> Using SuperLoader!!!!!")
            loader = SuperDataLoader(loader, 4, num_workers=1)
        return loader,dataset.count_video(),train_sampler
    else:
        loader = DataLoaderX(dataset,
                             batch_size=cfg.TRAIN.BATCH_SIZE,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=False
                             )
        if cfg.SUPER_LOADER:
            print("=> Using SuperLoader!!!!!")
            loader = SuperDataLoader(loader, 4, num_workers=1)
        return loader,dataset.count_video()

def val_loader_local(transform=None):
    dataset = val_base(transform=transform)
    # if cfg.USE_DISTRIBUTED:
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     loader = torch.utils.data.DataLoader(dataset,
    #                                         batch_size=cfg.VAL.BATCH_SIZE,
    #                                         num_workers=8,
    #                                         pin_memory=False,
    #                                         sampler=val_sampler)
    #
    #     return loader,val_sampler
    # else:
    #     loader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=cfg.VAL.BATCH_SIZE,
    #                                          shuffle=False,
    #                                          num_workers=8,
    #                                          pin_memory=False)
    #
    #     return loader
    loader = DataLoaderX(dataset,
                         batch_size=cfg.VAL.BATCH_SIZE,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=False)
    return loader,dataset.count_video()

def train_loader_remote_dpflow(transform=None):
    dataset = train_base(transform=transform)
    dpflow_addr_data = GenerateDPFlowAddr('data', 'global', file_name="./data_provider.py", timestamp='local')
    print("===== Data DPFlow addr is {} =====".format(dpflow_addr_data))
    loader = DPFlowDataLoaderDP(dataset,
                                cfg.REDIS_TRAIN_BATCH_SIZE,
                                dpflow_addr_data,
                                4,
                                num_workers=1)
    return loader,dataset.count_video()
def val_loader_remote_dpflow():
    dataset = val_base(transform=transform)
    loader = DataLoaderX(dataset,
                         batch_size=cfg.REDIS_VAL_BATCH_SIZE,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=False)
    return loader,dataset.count_video()

def train_loader_remote_pipe(transform=None):
    dataset = train_base(transform=transform)
    loader = PipeDataLoaderDP(dataset,
                              cfg.TRAIN.BATCH_SIZE,
                              8,
                              num_workers=1)
    return loader,dataset.count_video()


if __name__ == '__main__':
    # uint test
    crop_size = 224
    backbone = 'BNInception'
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    transform = torch_transforms.Compose([
                        GroupMultiScaleCrop(crop_size, [1, .875, .75, .66]),
                        GroupRandomHorizontalFlip(is_flow=False),
                        Stack(roll=(backbone == 'BNInception')),
                        ToTorchFormatTensor(div=(backbone != 'BNInception')),
                        GroupNormalize(input_mean, input_std)])

    # cfg.DATASET.LOAD_METHOD = 'TSNDataSet'
    # cfg.TRAIN.SEG_NUM = 5
    # cfg.TRAIN.BASE_BATCH_SIZE = 8
    # cfg.SUPER_LOADER = False

    # Test train_loader_local  TODO: 速度Rank-4, 覆盖率Rank-1
    loader,_ = train_loader_local(transform=transform)

    # Test superdataloader TODO: 速度Rank-2, 覆盖率Rank-3
    # cfg.SUPER_LOADER = True
    # loader = train_loader_local(transform=transform)


    # # Test train_loader_remote_dpflow TODO: 速度Rank-1, 覆盖率Rank-4
    # loader = train_loader_remote_dpflow(transform=transform)

    # # Test train_loader_remote_pipe TODO: 速度Rank-3, 覆盖率Rank-2
    # loader = train_loader_remote_pipe(transform=None)

    print("=> Loader length is %d"%(len(loader)))
    cnt = 0
    batch_time = AverageMeter()
    s = time.time()
    t0 = time.time()
    for epoch in range(1):
        iter_loader = iter(loader)
        index_lst = []
        for i in range(len(loader)):
            print(i)
            (index,data,label) = next(iter_loader)
            index_lst.extend(index.tolist())
            print(data.shape)
            batch_time.update(time.time() - t0)
            print("=>epoch %d [%d] step time is %s" % (epoch, i, str(batch_time.val)))
            print("=>epoch %d [%d] step avg time is %s" % (epoch, i, str(batch_time.avg)))
            t0 = time.time()
        print("index_lst len:",len(index_lst))
        print("set(index_lst) len:",len(set(index_lst)))
    e = time.time()
    print("=>One Epoch %.6f"%((e-s)))