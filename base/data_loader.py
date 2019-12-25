'''
Description: This file is used to load video from file system.
'''
import itertools
import os
import sys
import time
import random
import torch
import queue
import json
import cv2
import traceback
import numpy as np
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as torch_transforms
from prefetch_generator import BackgroundGenerator
from multiprocessing import Process, Queue,Lock
## self define file
from transform import *
from config import ActivityConfig as cfg
from utils import GenerateDPFlowAddr,AverageMeter,PipeInput
## DPFlow
if cfg.REDIS_MODE:
    import redis            ## Megvii
    import nori2 as nori    ## Megvii
    from dpflow import control,InputPipe ##Megvii

'''
Class Name: VideoRecord
Description: It's used to record video instance(item) info,such as seq_path,seq_len,seq_id_label
'''
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def seq_path(self):
        return self._data[0]

    @property
    def seq_len(self):
        return int(self._data[1])

    @property
    def seq_id_label(self):
        return int(self._data[2])
'''
Class Name: TSNDataSet
Description: It's used to load video data from file system according to meta file of dataset.
'''
class TSNDataSet(data.Dataset):
    def __init__(self, video_path=None,
                dataset_type=None,
                meta_file_name=None,
                sample_method=None,
                seg_num_ext=None,
                mode=None,
                transform=None,
                modality=None,
                img_format=None):
        ## To initialize parameters
        self.video_path = video_path
        self.dataset_type = dataset_type
        self.meta_file_name = meta_file_name
        self.sample_method = sample_method
        self.seg_num_ext = seg_num_ext
        self.mode = mode
        self.transform = transform
        self.modality = modality
        self.img_format = img_format

        self.item_list = None
        self._param_check()     ## Checking parameters
        self._parse_meta_file() ## Parsing meta file of dataset

    def __getitem__(self, index):
        '''
        :param index: 'index' is used to load given video
        :return: index，video data，video label
        '''
        video_item = self.item_list[index]
        img_path = os.path.join(self.video_path, video_item.seq_path)
        ## To get index by sampler
        if self.sample_method == 'seg_random':
            _index = self._seg_random_sampler(video_item.seq_len, self.seg_num_ext, 0)
        if self.sample_method == 'seg_ratio':
            _index = self._seg_ratio_sampler(video_item.seq_len, self.seg_num_ext, 0, 0.5)
        ## To get data from file system by index
        data = []
        for i in range(self.seg_num_ext):
            data.extend(self._load_image(img_path, _index[i]))
        ## Data augmentation
        data = self.transform(data)
        return index,data, video_item.seq_id_label

    def __len__(self):
        return len(self.item_list)

    def get(self, index):
        return self.__getitem__(index)

    def length(self):
        return self.__len__()

    def _load_image(self,img_path,idx):
        '''
        :param img_path: The absolute path of frames of videos
        :param idx: The index of frames
        :return: The original data of frames
        '''
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(img_path, self.img_format.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(img_path, self.img_format.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(img_path, self.img_format.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _param_check(self):
        if self.video_path is None:
            raise ValueError("video_path is None")
        if not (self.dataset_type in ['video', 'image']):
            raise ValueError("dataset_type occurs error,dataset_type should be ['video','image']")
        if not (self.sample_method in ['seg_random', 'seg_ratio', 'seg_seg']):
            raise ValueError("just support sample method in ['seg_random','seg_ratio','seg_seg'] ")
        if not (self.mode in ['train', 'val', 'test']):
            raise ValueError("just support mode in ['train','val','test']")
        if self.meta_file_name is None:
            raise ValueError("meta_file_name is None")
        if self.seg_num_ext is None:
            raise ValueError("seg_num_ext is None")
        (_, ext) = os.path.splitext(self.meta_file_name)
        if not (ext in ['.pkl', '.csv']):
            raise ValueError("meta_file_name is not ['.pkl','.csv'] file")

    def _parse_meta_file(self):
        meta_file_path = self.meta_file_name
        (_, ext) = os.path.splitext(self.meta_file_name)
        if self.dataset_type == 'video':
            if ext == '.csv':
                self.item_list = [VideoRecord(x.strip().split(',')) for x in open(meta_file_path)]
                print("=> load %s number of videos is %d"%(self.mode,len(self.item_list)))

    def _seg_random_sampler(self, seq_len, seg_num, s):
        '''
        :param seq_len: The length of a video
        :param seg_num:  The video is divided seg_num segments
        :param s: The index of the first frame from videos
        :return: The list contains index set
        '''
        if seq_len < seg_num:
            raise ValueError("seq_len<seg_num", seq_len, seg_num)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + random.randint(0, seg_len - 1) for i in range(seg_num)]
        return index

    def _seg_ratio_sampler(self, seq_len, seg_num, s, ratio):
        '''
        :param seq_len: The length of a video
        :param seg_num:  The video is divided seg_num segments
        :param s: The index of the first frame from videos
        :param ratio: None
        :return: The list contains index set
        '''
        if seq_len < seg_num:
            raise ValueError("seq_len<seg_num", seq_len, seg_num)
        if not (ratio >= 0.0 and ratio <= 1.0):
            raise ValueError("0<=ratio<=1", ratio)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + int((seg_len - 1) * ratio) for i in range(seg_num)]
        return index

    def _seg_seg_sampler(self, seq_len, seg_num_ext, seg_num_inner, func=None, sampler_type=None, ratio=0.5):
        if seq_len < seg_num_ext:
            raise ValueError("seq_len<seg_num", seq_len, seg_num_ext)
        seg_len_ext = seq_len // seg_num_ext
        if seg_len_ext < seg_num_inner:
            raise ValueError("seg_len_ext<seg_num_inner", seg_len_ext, seg_num_inner)
        if func is None:
            raise ValueError("No define single sampler,func is None")
        if not (sampler_type in ['rand', 'ratio']):
            raise ValueError("No define sampler_type or sampler_type,type is error")
        index = []
        for i in range(seg_num_ext):
            s = i * seg_len_ext
            if sampler_type == 'rand':
                index += func(seg_len_ext, seg_num_inner, s)
            if sampler_type == 'ratio':
                index += func(seg_len_ext, seg_num_inner, s, ratio)
        return index
'''
Class Name: DataLoaderX
Description: It's used to accelerate Dataloader of Pytorh
Installation： pip3 install prefetch_generator
'''
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),max_prefetch=1)
'''
Description: It's used to make iterator can be cycled. _spl is a prefix from SuperDataLoader. 
'''
def _spl_cycle(iterable):
    while True:
        for x in iterable:
            yield x
'''
Description: It' used to set random seed. _spl is a prefix from SuperDataLoader.
'''
def _spl_seed(seed_id=42):
    # random.seed(seed_id)
    # np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    # torch.cuda.manual_seed(seed_id)
    # torch.cuda.manual_seed_all(seed_id)
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED']=str(seed_id)
'''
Description: It's a function which is used in MultiProcessing. It serves SuperDataLoader.
'''
def _spl_worker_loader(loader,lock,buffer_queue,worker_id=None):
    _spl_seed(worker_id)
    try:
        data_input = iter(_spl_cycle(loader))
        # data_input = itertools.islice(data_input,worker_id,None,2)
    except Exception:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        traceback.print_tb(exc_traceback_obj)
    while True:
        try:
            # lock.acquire()
            index, data, label = next(data_input)
            # lock.release()
        except Exception:
            print("Buffer[{}] Error occured while getting data from loader!".format(worker_id))
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            continue
        else:
            # print("Buffer[{}] Cnt [{}]!".format(worker_id,cnt))
            pass

        while True:
            try:
                buffer_queue.put((index, data, label), block=True, timeout=1)
            except queue.Full:
                continue
            except Exception:
                print("Buffer[{}] Error occured while putting data into buffer queue!".format(worker_id))
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)
                break
            else:
                # print("Buffer[{}] put [{}] batch data into buffer queue.".format(worker_id,cnt))
                break
'''
Class Name: SuperDataLoader
Description: It' used to accelerate DataLoader of Pytorch.
Notice: It may lead to data samples be getted duplicated in one epoch!!!.
'''
class SuperDataLoader(object):
    def __init__(self,loader,num_workers_reading_buffer,num_workers=1):
        self.loader = loader
        self.num_workers = num_workers
        self.num_workers_reading_buffer = num_workers_reading_buffer
        self.shutdown = False
        self.lock = Lock()
        self._start()
        # self.buffer = []

    # for inherit
    def _start(self):
        if self.num_workers_reading_buffer > 0:
            self._reading_process()

    def _reading_process(self):
        self.training_reading_buffer = Queue(maxsize=self.num_workers_reading_buffer)
        self.readers = []
        for i in range(self.num_workers_reading_buffer):
            w = Process(
                    target=_spl_worker_loader,
                    args=(
                        self.loader,
                        self.lock,
                        self.training_reading_buffer,
                        i,
                        )
                    )
            w.deamon = True
            w.start()
            self.readers.append(w)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            if self.num_workers_reading_buffer > 0:
                # self.training_reading_buffer.close()
                for w in self.readers:
                    w.terminate()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

    def __next__(self):
        while True:
            try:
                # batch = self.training_reading_buffer.get_nowait()
                batch = self.training_reading_buffer.get(block=True, timeout=1)
            except queue.Empty:
                continue
            except Exception:
                print(sys.exc_info())
                continue
            else:
                return batch
'''
Description: It's used to load traning data.It support DataLoaderX,SuperDataLoader.
'''
def train_dataloader(transform=None):
    dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                        dataset_type='video',
                        meta_file_name=cfg.DATASET.TRAIN_META_PATH,
                        sample_method=cfg.DATASET.TRAIN_SAMPLE_METHOD,
                        seg_num_ext=cfg.TRAIN.SEG_NUM,
                        mode='train',
                        transform=transform,
                        modality=cfg.TRAIN.MODALITY,
                        img_format=cfg.DATASET.IMG_FORMART)

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
        return loader,train_sampler
    else:
        loader = DataLoaderX(dataset,
                                             batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=8,
                                             pin_memory=False
                                             )
        if cfg.SUPER_LOADER:
            print("=> Using SuperLoader!!!!!")
            loader = SuperDataLoader(loader,8,num_workers=1)
        return loader
'''
Description: It's used to load validation data.It support DataLoaderX.
'''
def val_dataloader(transform=None):
    dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                        dataset_type='video',
                        meta_file_name=cfg.DATASET.VAL_META_PATH,
                        sample_method=cfg.DATASET.VAL_SAMPLE_METHOD,
                        seg_num_ext=cfg.VAL.SEG_NUM,
                        mode='val',
                        transform=transform,
                        modality=cfg.VAL.MODALITY,
                        img_format=cfg.DATASET.IMG_FORMART)
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

    return loader
##TODO: Pipe Loader
def _worker_batch_receiver_pipe_dp(lock,buffer_queue,worker_id=None):
    _in = PipeInput(worker_id)
    while True:
        try:
            # lock.acquire()
            d = _in.get()
            idx = d['id']
            index = d['index']
            data = d['data']
            label = d['label']
            batch_samples = torch.from_numpy(data)
            batch_labels = torch.from_numpy(label)
            batch_indexs = torch.from_numpy(index)
            # lock.release()
        except Exception:
            print("Buffer[{}] Error occured while getting data from DPFlow!".format(worker_id))
            continue
        else:
            pass
            # print("Buffer[{}] got batch data from DPFlow".format(worker_id))

        while True:
            try:
                buffer_queue.put((batch_indexs,batch_samples,batch_labels), block=True, timeout=1)
            except queue.Full:
                continue
            except Exception:
                print("Buffer[{}] Error occured while putting data into buffer queue! idx: {}.".format(worker_id, idx))
                print(sys.exc_info())
                break
            else:
                # print("Buffer[{}] put batch data into buffer queue.".format(worker_id))
                break

class PipeDataLoaderDP(object):
    def __init__(self,  dataset,
                        batch_size,
                        num_workers_reading_buffer,
                        num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_workers_reading_buffer = num_workers_reading_buffer
        self.shutdown = False
        self.lock = Lock()

        self._start()

    # for inherit
    def _start(self):
        if self.num_workers_reading_buffer > 0:
            self._reading_process()

    def _reading_process(self):
        self.training_reading_buffer = Queue(maxsize=self.num_workers_reading_buffer)
        self.readers = []

        for i in range(self.num_workers_reading_buffer):
            w = Process(
                    target=_worker_batch_receiver_pipe_dp,
                    args=(
                        self.lock,
                        self.training_reading_buffer,
                        i, )
                    )
            w.deamon = True
            w.start()
            self.readers.append(w)

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __iter__(self):
        return self

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True

            if self.num_workers_reading_buffer > 0:
                # self.training_reading_buffer.close()
                for w in self.readers:
                    w.terminate()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

    def __next__(self):
        while True:
            try:
                batch = self.training_reading_buffer.get(block=True, timeout=1)
            except queue.Empty:
                continue
            except Exception:
                print(sys.exc_info())
                continue
            return batch

def train_pipe_dataloader_dp(transform=None):
    dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                         dataset_type='video',
                         meta_file_name=cfg.DATASET.TRAIN_META_PATH,
                         sample_method=cfg.DATASET.TRAIN_SAMPLE_METHOD,
                         seg_num_ext=cfg.TRAIN.SEG_NUM,
                         mode='train',
                         transform=transform,
                         modality=cfg.TRAIN.MODALITY,
                         img_format=cfg.DATASET.IMG_FORMART)
    loader = PipeDataLoaderDP(dataset,
                       cfg.TRAIN.BATCH_SIZE,
                       8,
                       num_workers=1)
    return loader
##TODO: End Pipe

##TODO: DPFlowDataset,DPFlowDataLoaderDP
'''
Class Name: DPFlowDataset
Description: To get data by nori,redis
'''
class DPFlowDataset(data.Dataset):
    def __init__(self,dataset_name=None,
                      mode=None,
                      transformer=None,
                      sample_method=None,
                      seg_num_ext=None):
        self.dataset_name = dataset_name
        self.mode = mode  # 'training' 'validation'
        self.transformer = transformer
        self.sample_method = sample_method
        self.seg_num_ext = seg_num_ext

        self.nori_fetcher = nori.Fetcher()
        self._init_redis()

    def __getitem__(self, index):
        assert index < len(self), \
            'index({}) should be less than samples({})'.format(index, len(self))
        # nid_lists, label, seg_name = self._redis_get(index)
        while True:
            try:
                nid_lists, label, seg_name = self._redis_get(index)
            except redis.exceptions.ConnectionError as err:
                print('=> Failed getting index {} from redis due to {}! Retry after 1s...'.format(index, err))
                time.sleep(1)
                continue
            except ValueError as err:
                print('=>Failed getting index {} from redis due to unexpected error: {}! Will return None.' \
                      .format(index, err))
                return
            except IndexError as err:
                print('=>Failed getting index {} from redis due to unexpected error: {}! Retry after 1s...' \
                      .format(index, err))
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)
                time.sleep(1)
                continue
            except Exception as err:
                print('=>Failed getting index {} from redis due to unexpected error: {}! Will return None.' \
                      .format(index, err))
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)
                return
            else:
                break

        data = self._nid_parse(nid_lists, seg_name)
        data = self.transformer(data)
        if self.mode == 'training':
            data = data.unsqueeze(0)
            data = data.numpy()
        else:
            pass
        return (index,data,label)

    def __len__(self):
        return self._redis_get_atom('{}_{}_len'.format(self.dataset_name, self.mode), 'int')

    def _init_redis(self):
        pickle_path = cfg.REDIS_PATH
        dataset_redis_path = os.path.join(pickle_path, '{}_{}.redis_info'.format( self.dataset_name,  self.mode))
        if not os.path.exists(dataset_redis_path):
            print("=> %s don't exist!!!,please checking it"%(dataset_redis_path))
            sys.exit(0)

        with open(dataset_redis_path) as jsonf:
            redis_info = json.load(jsonf)
            self.redis_info = redis_info
            # print('=>redis_info {}'.format(redis_info))
            self.rs = redis.StrictRedis(host=self.redis_info['host'], port=self.redis_info['port'])

    def _redis_get_atom(self, key, var_type):
        if var_type == 'int':
            return int(self.rs.get(key).decode('utf-8'))
        elif var_type == 'str':
            return str(self.rs.get(key).decode('utf-8'))
        else:
            raise RuntimeError('unknow type {}'.format(var_type))

    def _redis_get(self, index):
        seg_name = self._redis_get_atom('{}_{}_{}'.format(self.dataset_name, self.mode, index), 'str')
        # print('Sample segment: {}'.format(seg_name))
        try:
            label = self._redis_get_atom('{}_label'.format(seg_name), 'int')
        except redis.exceptions.ConnectionError as err:
            raise err
        except Exception:
            label = self._redis_get_atom('{}_label'.format(seg_name), 'str')

        nr_frame = self._redis_get_atom('{}_len'.format(seg_name), 'int')

        if self.sample_method == 'seg_random':
            frame_idxs = self._seg_random_sampler(nr_frame,self.seg_num_ext,0)

        if self.sample_method == 'seg_ratio':
            frame_idxs = self._seg_ratio_sampler(nr_frame, self.seg_num_ext, 0,0.5)

        nid_lists = [self._redis_get_atom('{}_{}'.format(seg_name,frame_idx), 'str')for frame_idx in frame_idxs]

        return nid_lists, label, seg_name

    def _nid_parse(self, nid_list, seg_name):
        frames = []
        for _id in nid_list:
            while True:
                try:
                    _string = self.nori_fetcher.get(_id)
                    _numpy = np.frombuffer(_string, dtype=np.uint8)
                    _cv2image = cv2.imdecode(_numpy, cv2.IMREAD_COLOR)[:, :, ::-1]
                    _PILimage = [Image.fromarray(_cv2image)]
                    frames.extend(_PILimage)
                except Exception as err:
                    print('=> Segment {} nori_id {} parse error due to {}'.format(seg_name, _id, err))
                    continue
                else:
                    break
        return frames

    def get(self, index):
        return self.__getitem__(index)

    def length(self):
        return self.__len__()

    def _seg_random_sampler(self, seq_len, seg_num, s):
        if seq_len < seg_num:
            raise ValueError("seq_len<seg_num", seq_len, seg_num)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + random.randint(0, seg_len - 1) for i in range(seg_num)]
        return index

    def _seg_ratio_sampler(self, seq_len, seg_num, s, ratio):
        if seq_len < seg_num:
            raise ValueError("seq_len<seg_num", seq_len, seg_num)
        if not (ratio >= 0.0 and ratio <= 1.0):
            raise ValueError("0<=ratio<=1", ratio)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + int((seg_len - 1) * ratio) for i in range(seg_num)]
        return index

    def _seg_seg_sampler(self, seq_len, seg_num_ext, seg_num_inner, func=None, sampler_type=None, ratio=0.5):
        if seq_len < seg_num_ext:
            raise ValueError("seq_len<seg_num", seq_len, seg_num_ext)
        seg_len_ext = seq_len // seg_num_ext
        if seg_len_ext < seg_num_inner:
            raise ValueError("seg_len_ext<seg_num_inner", seg_len_ext, seg_num_inner)
        if func is None:
            raise ValueError("No define single sampler,func is None")
        if not (sampler_type in ['rand', 'ratio']):
            raise ValueError("No define sampler_type or sampler_type is error type")
        index = []
        for i in range(seg_num_ext):
            s = i * seg_len_ext
            if sampler_type == 'rand':
                index += func(seg_len_ext, seg_num_inner, s)
            if sampler_type == 'ratio':
                index += func(seg_len_ext, seg_num_inner, s, ratio)
        return index
'''
Description: It's used to get data from DPFlow'InputPipe. It serves DPFlowDataLoaderDP
'''
def _worker_batch_receiver_dp(dpflow_addr_data,buffer_queue,worker_id=None):
    if isinstance(dpflow_addr_data, str):
        d_input = InputPipe(dpflow_addr_data, buffer_size=1)
        print('=> new iq {}'.format(d_input))
        with control(io=[d_input]):
            try:
                data_input = iter(d_input)
            except Exception:
                print(sys.exc_info)
            while True:
                try:
                    idx,index,data,label = next(data_input)
                    batch_samples = torch.from_numpy(data)
                    batch_labels = torch.from_numpy(label)
                    batch_indexs = torch.from_numpy(index)
                except Exception:
                    print("Buffer[{}] Error occured while getting data from DPFlow!".format(worker_id))
                    continue
                else:
                    pass
                    # print("Buffer[{}] got batch data from DPFlow".format(worker_id))

                while True:
                    try:
                        buffer_queue.put((batch_indexs,batch_samples,batch_labels), block=True, timeout=1)
                    except queue.Full:
                        continue
                    except Exception:
                        print("Buffer[{}] Error occured while putting data into buffer queue! idx: {}.".format(worker_id, idx))
                        print(sys.exc_info())
                        break
                    else:
                        # print("Buffer[{}] put batch data into buffer queue.".format(worker_id))
                        break

'''
Class Name: DPFlowDataLoaderDP
Description: To get data by DPFlow. It's used in training.
'''
class DPFlowDataLoaderDP(object):
    def __init__(self,  dataset,
                        batch_size,
                        dpflow_addr_data,
                        num_workers_reading_buffer,
                        num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dpflow_addr_data = dpflow_addr_data
        self.num_workers = num_workers
        self.num_workers_reading_buffer = num_workers_reading_buffer

        self.shutdown = False
        self._start()

    # for inherit
    def _start(self):
        if self.num_workers_reading_buffer > 0:
            self._reading_process()

    def _reading_process(self):
        self.training_reading_buffer = Queue(maxsize=self.num_workers_reading_buffer)
        self.readers = []

        for i in range(self.num_workers_reading_buffer):
            w = Process(
                    target=_worker_batch_receiver_dp,
                    args=(
                        self.dpflow_addr_data,
                        self.training_reading_buffer,
                        i, )
                    )
            w.deamon = True
            w.start()
            self.readers.append(w)

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __iter__(self):
        return self

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True

            if self.num_workers_reading_buffer > 0:
                # self.training_reading_buffer.close()
                for w in self.readers:
                    w.terminate()

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

    def __next__(self):
        while True:
            try:
                batch = self.training_reading_buffer.get(block=True, timeout=1)
            except queue.Empty:
                continue
            except Exception:
                print(sys.exc_info())
                continue
            return batch
'''
Description: To provide training data by DPFlow
'''
def train_dpflow_dataloader_dp(transform=None):
    dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                         dataset_type='video',
                         meta_file_name=cfg.DATASET.TRAIN_META_PATH,
                         sample_method=cfg.DATASET.TRAIN_SAMPLE_METHOD,
                         seg_num_ext=cfg.TRAIN.SEG_NUM,
                         mode='train',
                         transform=transform,
                         modality=cfg.TRAIN.MODALITY,
                         img_format=cfg.DATASET.IMG_FORMART)
    # dataset = DPFlowDataset(dataset_name=cfg.REDIS_DATASET,
    #                         mode='training',  # validation
    #                         transformer=transform,
    #                         sample_method=cfg.REDIS_TRAIN_SAMPLE_METHOD,
    #                         seg_num_ext=cfg.REDIS_TRAIN_SEG_NUM)
    dpflow_addr_data = GenerateDPFlowAddr('data', 'global', file_name="./data_provider.py", timestamp='local')
    print("===== Data DPFlow addr is {} =====".format(dpflow_addr_data))
    loader = DPFlowDataLoaderDP(dataset,
                       cfg.REDIS_TRAIN_BATCH_SIZE,
                       dpflow_addr_data,
                       4,
                       num_workers=1)
    return loader
'''
Description: To provide validation data by DPFlow
'''
def val_dpflow_dataloader_dp(transform=None):
    dataset = TSNDataSet(video_path=cfg.DATASET.VIDEO_SEQ_PATH,
                         dataset_type='video',
                         meta_file_name=cfg.DATASET.VAL_META_PATH,
                         sample_method=cfg.DATASET.VAL_SAMPLE_METHOD,
                         seg_num_ext=cfg.VAL.SEG_NUM,
                         mode='val',
                         transform=transform,
                         modality=cfg.VAL.MODALITY,
                         img_format=cfg.DATASET.IMG_FORMART)
    # dataset = DPFlowDataset(dataset_name=cfg.REDIS_DATASET,
    #                         mode='validation',  # validation
    #                         transformer=transform,
    #                         sample_method=cfg.REDIS_VAL_SAMPLE_METHOD,
    #                         seg_num_ext=cfg.REDIS_VAL_SEG_NUM)
    loader = DataLoaderX(dataset,
                         batch_size=cfg.REDIS_VAL_BATCH_SIZE,
                         shuffle=False,
                         num_workers=8,
                         pin_memory=False)
    return loader
## TODO: End DPFlow



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
    ## Test train_dataloader  TODO: 速度Rank-4, 覆盖率Rank-1
    loader = train_dataloader(transform=transform)

    # ## Test superdataloader TODO: 速度Rank-2, 覆盖率Rank-3
    # loader = train_dataloader(transform=transform)
    # loader = SuperDataLoader(loader, 8, num_workers=1)

    # ## Test train_dpflow_dataloader_dp TODO: 速度Rank-1, 覆盖率Rank-4
    # loader = train_dpflow_dataloader_dp(transform=transform)

    # ## Test train_pipe_dataloader_dp TODO: 速度Rank-3, 覆盖率Rank-2
    # loader = train_pipe_dataloader_dp(transform=None)

    print("=> Loader length is %d",len(loader))
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
            # print(data.shape)
            batch_time.update(time.time() - t0)
            print("=>epoch %d [%d] step time is %s" % (epoch, i, str(batch_time.val)))
            print("=>epoch %d [%d] step avg time is %s" % (epoch, i, str(batch_time.avg)))
            t0 = time.time()
        print("index_lst len:",len(index_lst))
        print("set(index_lst) len:",len(set(index_lst)))
    e = time.time()
    print("=>One Epoch %.6f"%((e-s)))






