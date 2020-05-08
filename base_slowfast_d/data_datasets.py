'''
Description: This file is used to load video from file system.
'''
import io
import os
import sys
import time
import random
import torch
import json
import cv2
import traceback
import numpy as np
import pickle as pkl
from PIL import Image
import torch.utils.data as data
## self define file
from transform import *
from config import ActivityConfig as cfg
## DPFlow
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
# TODO: TSNDataSet
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
        if video_item.seq_len < 1:
            video_item = self.item_list[index+1]
        img_path = os.path.join(self.video_path, video_item.seq_path)
        ## To get index by sampler
        if self.sample_method == 'seg_random':
             _index = self._dense_random_sampler(video_item.seq_len, self.seg_num_ext,64, 0)
            #_index = self._seg_random_sampler(video_item.seq_len, self.seg_num_ext, 0)
        if self.sample_method == 'seg_ratio':
             _index = self._dense_uniform_sampler(video_item.seq_len, self.seg_num_ext,64, 0)
            #_index = self._seg_ratio_sampler(video_item.seq_len, self.seg_num_ext, 0, 0.5)
        ## To get data from file system by index
        data = []
        for i in range(self.seg_num_ext):
            data.extend(self._load_image(img_path, _index[i]))
        ## Data augmentation
        data = self.transform(data)
        # data = data.unsqueeze(0) # TODO:DPFlow
        # data = data.numpy()
        return index,data, video_item.seq_id_label

    def __len__(self):
        return len(self.item_list)

    def get(self, index):
        return self.__getitem__(index)

    def length(self):
        return self.__len__()

    def count_video(self):
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
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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

    def _dense_random_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = random.randint(0, (seq_len - clip_len) - 1)
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]

    def _dense_uniform_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = (seq_len - clip_len) // 2
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]
import mc
# TODO: TSNDataSet_Memcache
'''
Class Name: TSNDataSet_Memcache
Description: It's used to load video data from file system according to meta file of dataset.
'''
class TSNDataSet_Memcache(data.Dataset):
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
        self.is_memcached = True
        self.initialized = False

        self.item_list = None
        self._param_check()     ## Checking parameters
        self._parse_meta_file() ## Parsing meta file of dataset

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = \
                "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = \
                "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True
        return

    def _pil_loader(self,img_str):
        buff = io.BytesIO(img_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return [img]
    def __getitem__(self, index):
        '''
        :param index: 'index' is used to load given video
        :return: index，video data，video label
        '''
        video_item = self.item_list[index]
        if video_item.seq_len < 1:
            video_item = self.item_list[index+1]
        img_path = os.path.join(self.video_path, video_item.seq_path)
        ## To get index by sampler
        if self.sample_method == 'seg_random':
             #_index = self._dense_random_sampler(video_item.seq_len, self.seg_num_ext,64, 0)
            _index = self._seg_random_sampler(video_item.seq_len, self.seg_num_ext, 0)
        if self.sample_method == 'seg_ratio':
            # _index = self._dense_uniform_sampler(video_item.seq_len, self.seg_num_ext,64, 0)
            _index = self._seg_ratio_sampler(video_item.seq_len, self.seg_num_ext, 0, 0.5)
        ## To get data from file system by index
        data = []
        for i in range(self.seg_num_ext):
            data.extend(self._load_image(img_path, _index[i]))
        ## Data augmentation
        data = self.transform(data)
        # data = data.unsqueeze(0) # TODO:DPFlow
        # data = data.numpy()
        return index,data, video_item.seq_id_label

    def __len__(self):
        return len(self.item_list)

    def get(self, index):
        return self.__getitem__(index)

    def length(self):
        return self.__len__()

    def count_video(self):
        return self.__len__()

    def _load_image(self,img_path,idx):
        '''
        :param img_path: The absolute path of frames of videos
        :param idx: The index of frames
        :return: The original data of frames
        '''



        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            filename = os.path.join(img_path, self.img_format.format(idx))
            # memcached
            if self.is_memcached:
                self._init_memcached()
                value = mc.pyvector()
                self.mclient.Get(filename, value)
                value_str = mc.ConvertBuffer(value)
                try:
                    img = self._pil_loader(value_str)
                except:  # noqa
                    raise Exception("Invalid file!", filename)
            else:
                img = [Image.open(filename).convert('RGB')]
            return img


            # return [Image.open(os.path.join(img_path, self.img_format.format(idx))).convert('RGB')]
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
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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

    def _dense_random_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = random.randint(0, (seq_len - clip_len) - 1)
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]

    def _dense_uniform_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = (seq_len - clip_len) // 2
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]

##TODO: DPFlowDataset
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
        self.cls_dict = None
        self.buffer_nori_idx = None
        self.buffer_label = None
        self.buffer_seg_name = None
        self.nori_fetcher = nori.Fetcher()
        self._init_redis()

    def __getitem__(self, index):
        assert index < len(self), \
            'index({}) should be less than samples({})'.format(index, len(self))
        # nid_lists, label, seg_name = self._redis_get(index)
        while True:
            try:
                nid_lists, label, seg_name = self._redis_get(index)
                # if len(nid_lists) >= 9:
                #     print(index)
                #     print(seg_name)
                #     sys.exit(0)
                #     break

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
            pass
            # data = data.unsqueeze(0) # TODO:DPFlow
            # data = data.numpy()
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
        with open(os.path.join(cfg.REDIS_PATH,cfg.REDIS_LABEL_PKL),'rb') as f:
            self.cls_dict = pkl.load(f)

    def _redis_get_atom(self, key, var_type):
        if var_type == 'int':
            return int(self.rs.get(key).decode('utf-8'))
        elif var_type == 'str':
            return str(self.rs.get(key).decode('utf-8'))
        elif var_type == 'float':
            return float(self.rs.get(key).decode('utf-8'))
        else:
            raise RuntimeError('unknow type {}'.format(var_type))

    def _redis_get(self, index):
        seg_name = self._redis_get_atom('{}_{}_{}'.format(self.dataset_name, self.mode, index), 'str')
        # print('Sample segment: {}'.format(seg_name))
        try:
            label = self._redis_get_atom('{}_label'.format(seg_name), 'int')
            # label = self._redis_get_atom('{}_label'.format(seg_name), 'str')
            # label = self.cls_dict[label]
        except redis.exceptions.ConnectionError as err:
            raise err
        except Exception:
            label = self._redis_get_atom('{}_label'.format(seg_name), 'str')
            label = self.cls_dict[label]

        nr_frame = self._redis_get_atom('{}_len'.format(seg_name), 'int')
        if nr_frame < 1:
            print("=>nr_frame:", nr_frame)
            seg_name = self._redis_get_atom('{}_{}_{}'.format(self.dataset_name, self.mode, index+1), 'str')

            try:
                label = self._redis_get_atom('{}_label'.format(seg_name), 'int')
                # label = self._redis_get_atom('{}_label'.format(seg_name), 'str')
                # label = self.cls_dict[label]
            except redis.exceptions.ConnectionError as err:
                raise err
            except Exception:
                label = self._redis_get_atom('{}_label'.format(seg_name), 'str')
                label = self.cls_dict[label]

            nr_frame = self._redis_get_atom('{}_len'.format(seg_name), 'int')

        if self.sample_method == 'seg_random':
            frame_idxs = self._dense_random_sampler(nr_frame,self.seg_num_ext,64,0)
            # frame_idxs = self._seg_random_sampler(nr_frame,self.seg_num_ext,0)

        if self.sample_method == 'seg_ratio':
            frame_idxs = self._dense_uniform_sampler(nr_frame,self.seg_num_ext,64,0)
            # frame_idxs = self._seg_ratio_sampler(nr_frame, self.seg_num_ext, 0,0.5)
        # if len(frame_idxs) >= 9:
        #     print(frame_idxs)
        #     sys.exit(0)
        # frame_idxs[0] = frame_idxs[0]+1
        nid_lists = [self._redis_get_atom('{}_{}'.format(seg_name,frame_idx), 'str')for frame_idx in frame_idxs]
        self.buffer_label = label
        self.buffer_nori_idx = nid_lists
        self.buffer_seg_name = seg_name
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

    def count_video(self):
        return self.__len__()

    def _seg_random_sampler(self, seq_len, seg_num, s):
        if seq_len < seg_num:
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                 index.extend([s+i for i in range(seq_len)])
            tail_index = [s+i for i in range(seg_num-len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + random.randint(0, seg_len - 1) for i in range(seg_num)]
        return index

    def _seg_ratio_sampler(self, seq_len, seg_num, s, ratio):
        if seq_len < seg_num:
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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

    def _dense_random_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = random.randint(0, (seq_len - clip_len) - 1)
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]

    def _dense_uniform_sampler(self,seq_len, frames, clip_len, s):
        if seq_len < frames:
            index = [i for i in range(seq_len)] + [seq_len - 1 for i in range(frames - seq_len)]
            return [s + i for i in index]
        if frames <= seq_len <= clip_len:
            index = [i for i in range(seq_len)]
            return [s + i for i in index][::(seq_len // frames)][:frames]
        s = (seq_len - clip_len) // 2
        index = [i for i in range(clip_len)]
        return [s + i for i in index][::(clip_len // frames)][:frames]

##TODO: DPFlowUntrimmedDataset
'''
Class Name: DPFlowUntrimmedDataset
'''
class DPFlowUntrimmedDataset(DPFlowDataset):
    def __init__(self,dataset_name=None,
                      mode=None,
                      transformer=None,
                      sample_method=None,
                      seg_num_ext=None):
        super(DPFlowUntrimmedDataset, self).__init__(dataset_name,mode,transformer,sample_method,seg_num_ext)
        self._generate_instance_list()

    def __getitem__(self, index):
        seg_name = self.instance_list[index][2]
        sub_seq_index = self.instance_list[index][0]
        sub_seq_len = len(sub_seq_index)
        sub_sample_index = self._seg_ratio_sampler(sub_seq_len, self.seg_num_ext, 0, 0.5)
        nori_list = [self._redis_get_atom('{}_{}'.format(seg_name, sub_seq_index[sample_idx]), 'str') for
                     sample_idx in sub_sample_index]
        data = self._nid_parse(nori_list, seg_name)
        data = self.transformer(data)
        # data = data.unsqueeze(0)
        return (data, self.instance_list[index][1], self.instance_list[index][3])

    def __len__(self):
        return len(self.instance_list)

    def _generate_instance_list(self):
        self.instance_list = []
        video_num = self._redis_get_atom('{}_{}_len'.format(self.dataset_name, self.mode), 'int')
        self.video_num = video_num
        print("=> %d videos will be processed" % (video_num))
        for id in range(video_num):
            # if id > 10:
            #     break
            seg_name = self._redis_get_atom('{}_{}_{}'.format(self.dataset_name, self.mode, id), 'str')
            seq_len = self._redis_get_atom('{}_len'.format(seg_name), 'int')
            instance_num = self._redis_get_atom('{}_instance_num'.format(seg_name), 'int')
            label_list = []
            for i in range(instance_num):
                label = self._redis_get_atom('{}_{}_label'.format(seg_name, i), 'str')
                label = self.cls_dict[label]
                label_list.append(label)
                # label_list.append(torch.ones(1,1)*label)
                # label_list.append(torch.ones(1,1) * label+1)
            one_hot_label = self.one_hot_embedding(label_list, 200)
            seq_fps = self._redis_get_atom('{}_fps'.format(seg_name), 'float')
            seq_duration = seq_len // int(seq_fps)
            index = [i for i in range(seq_len)]
            index_group = [index[i * (seq_len // seq_duration):(i + 1) * (seq_len // seq_duration)] for i in
                           range(seq_duration)]
            for x in index_group:
                self.instance_list.append((x, id, seg_name, one_hot_label))
        print("=> generate instance finished!!!!!")
        print("=> instance length %d" % (len(self.instance_list)))

    def one_hot_embedding(self,labels_list, num_classes):
        '''
        Embedding labels to one-hot.
        '''
        y = torch.eye(num_classes, device='cpu')  # [D,D]
        out = torch.zeros(num_classes)
        for label in labels_list:
              out +=y[label]
        return out # [N,D]

    def count_video(self):
        return self.video_num

## TODO: HumanactionDataset
#import meghair.utils.io as io
'''
Class Name: HumanactionDataset
'''
class HumanactionDataset(data.Dataset):
    def __init__(self,pkl_root_path=None,
                      pkl_name=None,
                      seg_num_ext=None,
                      transformer=None,
                      sample_method=None,
                      mode=None):
        self.pkl_root_path = pkl_root_path
        self.pkl_name = pkl_name
        self.seg_num_ext = seg_num_ext
        self.transformer = transformer
        self.sample_method = sample_method
        self.mode = mode

        self.nori_fetcher =  nori.Fetcher()

        self.instance_list = None
        self._load_pkl_data()
    def _load_pkl_data(self):
        # print(self.pkl_root_path)
        # print(self.pkl_name)
        _path = os.path.join(self.pkl_root_path,self.pkl_name)
        if not os.path.exists(_path):
            print("=> pkl %s don't exist!!"%(_path))
            sys.exit(0)
        self.instance_list = io.load(_path)
        print("=> instance list len %d"%(len(self.instance_list.keys())))

    def _prepare_data(self):
        pass

    def __getitem__(self, index):
        ## get label
        if self.mode == 'training':
            tag_name = 'labels'
        else:
            tag_name = 'attributes'
        label_list = []
        sorted_keys = sorted(self.instance_list.keys())
        for x in cfg.HUMANACTION_LABEL_LIST:
            if x in self.instance_list[sorted_keys[index]][tag_name]:
                # assert (x in self.instance_list[sorted_keys[index]]['attributes']), (index,self.instance_list[sorted_keys[index]]['attributes'])
                if self.instance_list[sorted_keys[index]][tag_name][x] == 1:
                    label_list.append(1)
                elif self.instance_list[sorted_keys[index]][tag_name][x] == 0:
                    label_list.append(0)
                else:
                    label_list.append(-1)
            else:
                label_list.append(0)
        ## get data
        # seq_len = len(self.instance_list[sorted_keys[index]]['nori_id_seq']) #
        if 'nori_id_seq_resized' in list(self.instance_list[sorted_keys[index]].keys()):
            nori_seq = self.instance_list[sorted_keys[index]]['nori_id_seq_resized']
        else:
            nori_seq = self.instance_list[sorted_keys[index]]['nori_id_seq']
        seq_len = len(nori_seq)
        if self.mode == 'training':
            nori_index = self._seg_random_sampler(seq_len, self.seg_num_ext, 0)
        else:
            nori_index = self._seg_ratio_sampler(seq_len, self.seg_num_ext, 0, 0.5)

        nori_list = [nori_seq[i] for i in nori_index]
        data = self._nid_parse(nori_list, index)
        data = self.transformer(data)
        # data = data.unsqueeze(0) #TODO:DPFlow
        # data = data.numpy()
        return index, data, np.array(label_list)

    def __len__(self):
        return len(self.instance_list.keys())

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

    def _seg_random_sampler(self, seq_len, seg_num, s):
        if seq_len < seg_num:
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
        seg_len = seq_len // seg_num
        index = [s + i * seg_len + random.randint(0, seg_len - 1) for i in range(seg_num)]
        return index

    def _seg_ratio_sampler(self, seq_len, seg_num, s, ratio):
        if seq_len < seg_num:
            index = []
            cycle_len = seg_num // seq_len
            for i in range(cycle_len):
                index.extend([s + i for i in range(seq_len)])
            tail_index = [s + i for i in range(seg_num - len(index))]
            index.extend(tail_index)
            return index
            # raise ValueError("seq_len<seg_num", seq_len, seg_num)
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

    def count_video(self):
        return self.__len__()






