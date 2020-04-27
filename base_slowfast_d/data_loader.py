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
import traceback
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from multiprocessing import Process, Queue,Lock
## self define file
from config import ActivityConfig as cfg
from utils import PipeInput

## TODO: DataLoaderX
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

## TODO: SuperDataLoader
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
##TODO: End Pipe

##TODO:DPFlowDataLoaderDP
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

## TODO: End DPFlow








