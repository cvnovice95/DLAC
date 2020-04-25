import os
import sys
import pprint as pp
import pickle as pkl
import torch
import shutil
import time
import math
import numpy as np
import hashlib
import getpass
import os
import sys
import traceback
import pickle as pkl
import torch.distributed as dist
from multiprocessing import Process,Lock
## self define file
from config import ActivityConfig as cfg
'''
Class Name: IO
'''
class IO(object):
    def __init__(self):
        pass
    @staticmethod
    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                continue
                # self.del_file(c_path)
            else:
                os.remove(c_path)
    def check_GPU(self):
        if torch.cuda.is_available():
            print("=> Yon can use CUDA!!!")
            if torch.cuda.device_count() > 1:
                print("=> Oh!!! You can use %d GPUs"%(torch.cuda.device_count()))
            else:
                print("=> %d GPUs can be used!!!"%(torch.cuda.device_count()))
        else:
            print("=> You don't use CUDA!")
            sys.exit(0)
    def check_folder(self):
        path_list = [cfg.SNAPSHOT_LOG,
                     cfg.SNAPSHOT_LOG_DEBUG,
                     cfg.SNAPSHOT_CHECKPOINT,
                     cfg.SNAPSHOT_SUMMARY,
                     cfg.SNAPSHOT_CONFIG,
                     cfg.PRETRAIN_MODEL_ZOO]
        print("checking folder......")
        for x in path_list:
            if os.path.exists(x):
                print("%s is existed"%(x))
            else:
                print("%s is not existed" % (x))
                print("=> %s will be created" % (x))
                os.makedirs(x)
        ## check pretrain_model_zoo
        if len(os.listdir(cfg.PRETRAIN_MODEL_ZOO)) == 0:
            print("%s is empty! Please add pretrain model!"%(cfg.PRETRAIN_MODEL_ZOO))
            sys.exit(0)
        print("checking dataset.....")
        path_list = [cfg.DATASET.VIDEO_SEQ_PATH,
                     cfg.DATASET.TRAIN_META_PATH,
                     cfg.DATASET.VAL_META_PATH]
        for x in path_list:
            if os.path.exists(x):
                print("%s is existed" % (x))
            else:
                print("%s is not existed" % (x))
                sys.exit(0)

    def save_checkpoint(self,state, is_best, epoch):
        '''
        :param state:  {'epoch':
                        'model_dict':
                        'optimizer_dict':
                        'amp_dict':
                        'best_prec1':
                        }
        :param is_best:
        :param epoch:
        :return:
        '''

        filename = os.path.join(cfg.SNAPSHOT_CHECKPOINT, "{}_epoch_{}_{}_{}.pth.tar".format(cfg.TIMESTAMP,
                                                                                      epoch,
                                                                                      'A' if cfg.USE_APEX else 'NONE',
                                                                                      'D' if cfg.USE_DISTRIBUTED else 'NONE'))
        torch.save(state, filename)
        if is_best:
            best_name = os.path.join(cfg.SNAPSHOT_CHECKPOINT, "{}_best_{}_{}.pth.tar".format(cfg.TIMESTAMP,
                                                                                             'A' if cfg.USE_APEX else 'NONE',
                                                                                             'D' if cfg.USE_DISTRIBUTED else 'NONE'))
            shutil.copyfile(filename, best_name)

    def load_checkpoint(self,model=None,optimizer=None,amp=None):
        if (not (model is None)) and (not (optimizer is None)):
            if cfg.RESUME_TYPE == 'resume':
                if cfg.TRAIN.RESUME:
                    _path = os.path.join(cfg.SNAPSHOT_CHECKPOINT, cfg.TRAIN.RESUME)
                    if os.path.exists(_path):
                        print("=> loading checkpoint %s" % (_path))
                        checkpoint = torch.load(_path)
                        cfg.TRAIN.START_EPOCH = checkpoint['epoch']
                        cfg.TRAIN.BEST_PREC = checkpoint['best_prec1']

                        own_state = model.state_dict()
                        for layer_name, param in checkpoint['model_dict'].items():
                            if 'module' in layer_name:
                                layer_name = layer_name[7:]
                            if isinstance(param, torch.nn.parameter.Parameter):
                                param = param.data

                            assert param.dim() == own_state[layer_name].dim(), \
                                '{} {} vs {}'.format(layer_name, param.dim(), own_state[layer_name].dim())
                            own_state[layer_name].copy_(param)

                        # model.load_state_dict(checkpoint['model_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_dict'])
                        if cfg.USE_APEX:
                            amp.load_state_dict(checkpoint['amp_dict'])
                        print("=> start epoch %d, best_prec1 %f"%(cfg.TRAIN.START_EPOCH,cfg.TRAIN.BEST_PREC))
                        print("=> loaded checkpoint epoch is %d" % (cfg.TRAIN.START_EPOCH))
                    else:
                        print("=> no checkpoint found! %s" % (_path))
                        sys.exit(0)

            if cfg.RESUME_TYPE == 'finetune':
                if cfg.TRAIN.RESUME:
                    _path = os.path.join(cfg.SNAPSHOT_CHECKPOINT, cfg.TRAIN.RESUME)
                    if os.path.exists(_path):
                        print("=> loading finetune model %s" % (_path))
                        finetune_model = torch.load(_path)
                        # cfg.TRAIN.START_EPOCH = checkpoint['epoch']
                        # cfg.TRAIN.BEST_PREC = checkpoint['best_prec']
                        # model.load_state_dict(checkpoint['state_dict'])
                        print("=> loaded finetune model")
                    else:
                        print("=> no finetune model found! %s" % (_path))
                        sys.exit(0)

    def save_config(self,content):
        if not cfg.KEEP_HISTORY:
            _path = os.path.join(cfg.SNAPSHOT_CONFIG)
            IO.del_file(_path)
        name = cfg.TIMESTAMP+"_cfg.txt"
        print("=>config %s will be saved"%(name))
        _path = os.path.join(cfg.SNAPSHOT_CONFIG,name)
        with open(_path,"w") as f:
            f.write(pp.pformat(content))
        name = cfg.TIMESTAMP + "_cfg.pkl"
        print("=>pkl config %s will be saved" % (name))
        _path = os.path.join(cfg.SNAPSHOT_CONFIG, name)
        with open(_path, "wb") as f:
            pkl.dump(content,f)
'''
Class Name: AverageMeter
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''
Class Name: Statistic
'''
class Statistic(object):
    class ClsInfo(object):
        def __init__(self):
            self.num = 0
            self.correct = 0
            self.error = 0
        @property
        def CntCorrect(self):
            self.correct +=1
            self.num += 1

        @property
        def CntError(self):
            self.error +=1
            self.num += 1


    def __init__(self,cls_num,pkl_name,txt_name,sample_num=None):
        self.cls_num = cls_num
        self.sample_num = sample_num
        self.ClsInfoList = [Statistic.ClsInfo() for i in range(cls_num)]
        self.error_sample_dict = {}
        self.sample_cnt = 0
        self.pkl_name = pkl_name
        self.txt_name = txt_name
    def add_error_sample(self,in_tuple):
        self.error_sample_dict[str(self.sample_cnt)]=in_tuple
    @property
    def cnt_sample(self):
        self.sample_cnt +=1
    def print_info(self,top_avg=None):
        f = open(self.txt_name,"w")
        print("\t\n=======Statistic Info==========\n")
        f.write("\t\n=======Statistic Info==========\n")
        for x in range(self.cls_num):
            print("Class Index: %d \n" % (x))
            f.write("Class Index: %d \n" % (x))
            if self.ClsInfoList[x].num != 0:
                print("Current Class Correct: %d\n"%(self.ClsInfoList[x].correct))
                f.write("Current Class Correct: %d\n"%(self.ClsInfoList[x].correct))
                print("Current Class Error: %d\n" % (self.ClsInfoList[x].error))
                f.write("Current Class Error: %d\n" % (self.ClsInfoList[x].error))
                print("Current Class Num: %d\n" % (self.ClsInfoList[x].num))
                f.write("Current Class Num: %d\n" % (self.ClsInfoList[x].num))
                print("Current Class Percent: %f\n"% (self.ClsInfoList[x].correct /self.ClsInfoList[x].num ))
                f.write("Current Class Percent: %f\n"% (self.ClsInfoList[x].correct /self.ClsInfoList[x].num ))
            else:
                print("Current Class Num is Zero!!!!")
                f.write("Current Class Num is Zero!!!!")
        print("\t\n=======End Statistic Info==========\n")
        f.write("\t\n=======End Statistic Info==========\n")
        print("error_sample length: %d "%(len(self.error_sample_dict)))
        f.write("error_sample length: %d "%(len(self.error_sample_dict)))
        print("top1 avg percent: %f "%(top_avg))
        f.write("top1 avg percent: %f "%(top_avg))
        f.close()
        with open(self.pkl_name,"wb") as f:
            pkl.dump(self.error_sample_dict,f)
        print("Done Statistic!!!!")

def accuracy(output, target, topk=(1,), s = None):
    """Computes the precision@k for the specified values of k"""
    # print(output.shape) [N,class_numm]
    # print(target.shape) [N]
    maxk = max(topk)      # topk (1,5) maxk = 5
    batch_size = target.size(0)  # batch_size = N

    _, pred = output.topk(maxk, 1, True, True)
    #print(pred.shape) #[N,5]
    pred = pred.t()  # [5,N]
    # t = target.view(1, -1) [1,N]
    # t = target.view(1, -1).expand_as(pred) [5,N]
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #[5,N]
    if not (s is None):
        for i in range(batch_size):
            s.cnt_sample
            if correct[0][i] == 1:
                s.ClsInfoList[target[i]].CntCorrect
            if correct[0][i] == 0:
                s.ClsInfoList[target[i]].CntError
                s.add_error_sample((pred[0][i].cpu().detach().numpy(),target[i].cpu().detach().numpy()))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_topK(matrix, K, axis=0):
    ind = np.argsort(-matrix, axis=axis,)
    full_sort = np.take_along_axis(matrix, ind, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)

def get_score_untrimmed(pred_score_list,id,class_num=None):
    if len(pred_score_list) == 0:
        print("=> [{}] pred_score_list is empty,please checking it.".format(id))
        sys.exit(0)
    # print("=> pred_score_list len %d"%(len(pred_score_list)))
    # pred_score_list [(1,class_num),...]
    def max_window_score(pred_score_list,w_size,class_num):
        stride = int(math.ceil(w_size*0.8))
        w_num = len(pred_score_list)//stride
        start_list = [i*stride for i in range(w_num)]
        w_list = []
        for x in start_list:
            item_list = []
            for i in range(w_size):
                if x + i >= len(pred_score_list):
                    item_list.append(pred_score_list[x + i-1])
                    break
                item_list.append(pred_score_list[x + i])
            # print("=> len",len(item_list))
            # print("=> w size",w_size)
            # print("=> x",x)
            # if len(item_list) == 0:
            #     print("++++")
            #     print(start_list)
            w = np.concatenate(item_list,axis=0)
            max_w = np.max(w, axis=0, keepdims=True)
            w_list.append(max_w)
        if len(w_list) == 0:
            res =  np.concatenate(pred_score_list,axis=0)
            return res,min(15,1)
        res = np.concatenate(w_list,axis=0)
        # res (w_num, class_num)
        if w_num//4 == 0:
            k = 1
        else:
            k = w_num//4
        return res,min(15,k)

    def top_k_pooling(matrix,K=None):
        topK = get_topK(matrix, K, axis=0)
        return np.mean(topK, axis=0, keepdims=True)
    multi_scale_pred_score = []
    for w_size in [1,2,4,8,16]:
        res,K = max_window_score(pred_score_list, w_size, class_num)
        pred = top_k_pooling(res,K=K)
        # pred (1,class_num)
        multi_scale_pred_score.append(pred)

    res = np.mean(np.concatenate(multi_scale_pred_score,axis=0), axis=0, keepdims=True)
    # res (1,class_num)
    return res

'''
some function which is used in distributed training
'''
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def GenerateDPFlowAddr(prefix, rank, file_name='exps/examples/main.py', timestamp='local'):
    """make a unique servable name.

    .. note::
        The resulting servable name is composed by the content of
        dependency files and the original dataset_name given.

    :param dataset_name: an datasets identifier, usually the argument
        passed to datasets.py:get
    :type dataset_name: str

    :param dep_files: files that the constrution of the datasets depends on.
    :type dep_files: list of str
    """

    suffix = str(rank)

    def md5sum(inputs):
        """md5 checksum of given string
        :return: md5 hex digest string
        """
        md5s = hashlib.md5()
        md5s.update(inputs)
        return md5s.hexdigest()

    parts = []
    with open(file_name, 'rb') as fdes:
        parts.append(md5sum(fdes.read()))
    return '{}.{}:{}.{}.{}'.format(
        prefix,
        getpass.getuser(),
        '.'.join(parts),
        timestamp,  # contains dataset name and phase and time
        suffix
    )
'''
Class Name: PipeOutput
'''
class PipeOutput(object):
    def __init__(self,name=None):

        if name is None:
            self.out_path = "/tmp/server_in.pipe"
        else:
            self.out_path =  "/tmp/server_in_"+str(name)+".pipe"
        if os.path.exists(self.out_path):
            print("=> Pipe %s has existed!"%(self.out_path))
        else:
            try:
                os.mkfifo(self.out_path)
            except Exception:
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)

        try:
            self.handle_send = os.open(self.out_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)
        except Exception:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)

    def _depack_data(self,data):
        o_data = None
        try:
            o_data = pkl.loads(data)
        except Exception:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            # print(o_data)
        return o_data

    def _pack_data(self,data):
        p_data = pkl.dumps(data)
        return p_data

    def put(self,data):
        p_data = self._pack_data(data)
        print(type(p_data))
        print("Len:",len(p_data))
        os.write(self.handle_send,p_data)
        # time.sleep(1)

    def close(self):
        os.close(self.handle_send)
'''
Class Name: PipeInput
'''
class PipeInput(object):
    def __init__(self,name=None):

        if name is None:
            self.in_path = "/tmp/server_in.pipe"
        else:
            self.in_path = "/tmp/server_in_"+str(name)+".pipe"
        if os.path.exists(self.in_path) :
            print("=> Pipe %s has existed!" % (self.in_path))
        else:
            try:
                os.mkfifo(self.in_path)
            except Exception:
                exc_type, exc_value, exc_traceback_obj = sys.exc_info()
                traceback.print_tb(exc_traceback_obj)
        try:
            self.handle_receiver = os.open(self.in_path, os.O_RDONLY)
        except Exception:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)

    def _depack_data(self,data):
        o_data = None
        try:
            o_data = pkl.loads(data)
        except Exception:
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
        return o_data

    def get(self):
        while True:
            data = os.read(self.handle_receiver,308281524)
            if len(data) == 0:
                continue
            else:
                print("len:",len(data))
                break
        o_data = self._depack_data(data)
        return o_data

    def close(self):
        os.close(self.handle_receiver)



