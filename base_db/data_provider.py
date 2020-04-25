import sys
import time
import queue
import random
import traceback
import numpy as np
from multiprocessing import Process, Queue, Value,Lock

## self define file
from model import Model
from config import ActivityConfig as cfg
from utils import GenerateDPFlowAddr,PipeOutput
from data  import train_provider_remote

## DPFlow

'''
Class Name: IndexSampler
Description: It's used to product index set of a batch randomly
'''
class IndexSampler(object):

    def __init__(self,length = None,batch_size=None):
        if length is None:
            raise ValueError("length is None")
        if batch_size is None:
            raise ValueError("bacth_size is None")
        self.length = length
        self.batch_size = batch_size
        self.index_list = [i for i in range(self.length)]
        random.shuffle(self.index_list)
        self.cnt = -1

    def __next__(self):
        while True:
            if (self.cnt) < (self.length//self.batch_size):
                self.cnt = self.cnt+1
                return [x for x in self.index_list[self.cnt*self.batch_size:(self.cnt+1)*self.batch_size]]
            else:
                return None
                # raise StopIteration
    def __len__(self):
        return (self.length//self.batch_size)+1

    def __iter__(self):
        return self

    def set_epoch(self):
        self.cnt = -1
        random.shuffle(self.index_list)
'''
Description: It's used to generate index set of a batch.
'''
def _worker_index_generator_dp(sampler,index_queue,send_idx):
    '''
    generate index (batch size)
    :param sampler: according to dataset size, sample index(batch size), output [i,...,j]
    :param index_queue: index buffer queue
    :param send_idx:
    :return:
    '''
    epoch  = 0
    print('=> Sampler Start to produce epoch {}.'.format(epoch))
    sampler.set_epoch()
    instance_iter = iter(sampler)
    while True:
        try:
            indice = next(instance_iter)
            if indice is None:
                epoch = 0 if epoch>10000 else epoch+1
                print('=> Sampler Start to produce epoch {}.'.format(epoch))
                sampler.set_epoch()
                instance_iter = iter(sampler)
                send_idx.value = 0
                continue
            while True:
                try:
                    index_queue.put((send_idx.value, indice))
                except queue.Full:
                    continue
                else:
                    if send_idx.value % 1 == 0:
                        print('=> Epoch: [{}] Put {}th index into index_queue.'.format(epoch, send_idx.value))
                    send_idx.value += 1
                    break
        except Exception as err:
            print('=> Unexpected error: {}'.format(err))
            break
'''
Description: It's used to send a batch data into DPFlow's OutputPipe
'''
def _worker_batch_sender_dp(dataset,index_queue,pipe_name,batch_size,worker_id=None):
    try:
        d_queue = OutputPipe(pipe_name, buffer_size=batch_size*4)
        with control(io=[d_queue]):
            ## Get Index from Index Generator
            while True:
                try:
                    r = index_queue.get(timeout=1)
                except queue.Empty:
                    continue
                except Exception:
                    print("=> Worker[{}] Error occured while getting index from index queue!".format(worker_id))
                    break

                idx, instance_idx = r
                # print(instance_idx)
                print("=> Worker[{}] Got {}th idx from index_queue.".format(worker_id, idx))
                try:
                    batch_data = []
                    label_data = []
                    index_data = []
                    for x in instance_idx:
                        # print(dataset.get(x).shape)
                        index,data_item,label_item = dataset.get(x)
                        if data_item is None:
                            print("=>Worker[{}] Got None data in {}th idx.".format(worker_id, idx))
                            continue
                        batch_data.append(data_item)
                        label_data.append(label_item)
                        index_data.append(index)
                    if not (len(batch_data)== batch_size):
                        continue
                    raw_sample = np.concatenate(batch_data,axis=0)
                    label_data = np.array(label_data)
                    index_data = np.array(index_data)
                    # print(raw_sample.shape)
                    # print(label_data.shape)
                except Exception:
                    exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                    traceback.print_tb(exc_traceback_obj)
                    print("Worker:[{}] Error occured while getting {}th batch from datasets!".format(worker_id, idx))
                    print(raw_sample.shape)
                    break
                else:
                    d_queue.put_pyobj((idx,index_data,raw_sample,label_data))
                    print("worker[{}] put {}th batch idx into DPFlow.".format(worker_id, idx))
                    del raw_sample
    except KeyboardInterrupt:
        pass
    finally:
        print('Worker: [{}] exiting.'.format(worker_id))
'''
Class Name: DPFlowProviderDP
Description: It's used to send batch data by multiprocess. In here, it needs function mainly as follows:
_worker_index_generator_dp()
_worker_batch_sender_dp()
'''
class DPFlowProviderDP(object):
    def __init__(self,dataset,sampler,dpflow_addr_data,num_workers,batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.dpflow_addr_data = dpflow_addr_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.index_queue = Queue(maxsize=4*num_workers)
        self.send_idx = Value('i', 0)
        self.shutdown = False

    def start_to_provide(self):
        self._start_providing_index()
        self._start_providing_dpflow()

    def _start_providing_index(self):
        self.index_worker = Process(
            target= _worker_index_generator_dp,
            args=(
                self.sampler,
                self.index_queue,
                self.send_idx,
            )
        )
        self.index_worker.daemon = False
        self.index_worker.start()
        while self.index_queue.empty():
            continue

    def _start_providing_dpflow(self):
        self.dpflow_workers = []
        for i in range(self.num_workers):
            w = Process(
                target=_worker_batch_sender_dp,
                args=[
                    self.dataset,
                    self.index_queue,
                    self.dpflow_addr_data,
                    self.batch_size,
                    i,
                ]
            )
            w.daemon = False
            w.start()
            self.dpflow_workers.append(w)

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.index_worker.terminate()
            for w in self.dpflow_workers:
                w.terminate()
            # self.index_queue.cancel_join_thread()
            self.index_queue.close()
            # self.ctrl_epoch.stop()

    def __del__(self):
        self._shutdown_workers()

'''
Description: It's used to send data into pipe. It serves PipeProviderDP.
'''
def _worker_batch_sender_pipe_dp(dataset,index_queue,lock,batch_size,worker_id=None):
    _out = PipeOutput(worker_id)
    try:
        ## Get Index from Index Generator
        while True:
            try:
                r = index_queue.get(timeout=1)
            except queue.Empty:
                continue
            except Exception:
                print("=> Worker[{}] Error occured while getting index from index queue!".format(worker_id))
                break

            idx, instance_idx = r
            # print(instance_idx)
            print("=> Worker[{}] Got {}th idx from index_queue.".format(worker_id, idx))
            try:
                batch_data = []
                label_data = []
                index_data = []
                for x in instance_idx:
                    # print(dataset.get(x).shape)
                    index,data_item,label_item = dataset.get(x)
                    if data_item is None:
                        print("=>Worker[{}] Got None data in {}th idx.".format(worker_id, idx))
                        continue
                    batch_data.append(data_item)
                    label_data.append(label_item)
                    index_data.append(index)
                if not (len(batch_data)== batch_size):
                    continue
                raw_sample = np.concatenate(batch_data,axis=0)
                label_data = np.array(label_data)
                index_data = np.array(index_data)
                # print(raw_sample.shape)
                # print(label_data.shape)
            except Exception:
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)
                print("Worker:[{}] Error occured while getting {}th batch from datasets!".format(worker_id, idx))
                print(raw_sample.shape)
                break
            else:
                d = {}
                d['id'] = idx
                d['index'] = index_data
                d['data'] = raw_sample
                d['label'] = label_data
                # lock.acquire()
                _out.put(d)
                # lock.release()
                print("worker[{}] put {}th batch idx into DPFlow.".format(worker_id, idx))
                del raw_sample
    except KeyboardInterrupt:
        pass
    finally:
        print('Worker: [{}] exiting.'.format(worker_id))
'''
Class Name: PipeProviderDP
Description: It's used to send batch data by multiprocess.
'''
class PipeProviderDP(object):
    def __init__(self,dataset,sampler,num_workers,batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.index_queue = Queue(maxsize=4*num_workers)
        self.send_idx = Value('i', 0)
        self.shutdown = False
        self.lock = Lock()

    def start_to_provide(self):
        self._start_providing_index()
        self._start_providing_dpflow()

    def _start_providing_index(self):
        self.index_worker = Process(
            target= _worker_index_generator_dp,
            args=(
                self.sampler,
                self.index_queue,
                self.send_idx,
            )
        )
        self.index_worker.daemon = False
        self.index_worker.start()
        while self.index_queue.empty():
            continue

    def _start_providing_dpflow(self):
        self.dpflow_workers = []
        for i in range(self.num_workers):
            w = Process(
                target=_worker_batch_sender_pipe_dp,
                args=[
                    self.dataset,
                    self.index_queue,
                    self.lock,
                    self.batch_size,
                    i,
                ]
            )
            w.daemon = False
            w.start()
            self.dpflow_workers.append(w)

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.index_worker.terminate()
            for w in self.dpflow_workers:
                w.terminate()
            # self.index_queue.cancel_join_thread()
            self.index_queue.close()
            # self.ctrl_epoch.stop()

    def __del__(self):
        self._shutdown_workers()

if __name__ == '__main__':
    ## ID
    dpflow_addr_data = GenerateDPFlowAddr('data', 'global', file_name="./data_provider.py", timestamp='local')
    print("===== Data DPFlow addr is {} =====".format(dpflow_addr_data))
    ## prepare transform
    _model = Model()
    net = _model.select_model('tsn')
    train_transform = net.train_transform()
    ## prepare dataset
    inst = train_provider_remote(train_transform)

    provider = DPFlowProviderDP(inst,
                     IndexSampler(length=inst.length(),batch_size=cfg.REDIS_TRAIN_BATCH_SIZE),
                     dpflow_addr_data,
                     8,
                     cfg.REDIS_TRAIN_BATCH_SIZE,
                     )

    # provider = PipeProviderDP(inst,
    #                           IndexSampler(length=inst.length(), batch_size=cfg.REDIS_TRAIN_BATCH_SIZE),
    #                           8,
    #                           cfg.REDIS_TRAIN_BATCH_SIZE)

    provider.start_to_provide()

    while True:
        time.sleep(5)

