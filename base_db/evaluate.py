import sys
import os
import time
import torch
import pprint as pp
import numpy as np
import pickle as pkl
##self define file
from model import Model
from config import ActivityConfig as cfg
from data import val_loader_local
from utils import IO,accuracy,AverageMeter,Statistic
if cfg.REDIS_MODE:
    from data import val_loader_remote_dpflow
def main():
    if 'E' not in cfg.EXP_TYPE:
        print("=> Your Exp Type is %s, not 'E' "%(cfg.EXP_TYPE))
        sys.exit(0)

    ## load checkpoint
    exp_name = "{}_{}_{}".format('T', cfg.MODEL_NAME, cfg.EXP_TAG)
    checkpoint_path = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', exp_name, 'checkpoint')
    config_path = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', exp_name, 'config')
    checkpoint_name = None

    if not os.path.exists(config_path):
        print("=> %s is not exists!!!!!!"%(config_path))
        sys.exit(0)
    if not os.path.exists(checkpoint_path):
        print("=> %s is not exists!!!!!!"%(checkpoint_path))
        sys.exit(0)

    lst = os.listdir(checkpoint_path)
    lst = sorted(lst)
    if len(lst) == 0:
        print("=> don't have any checkpoint file!!!")
        sys.exit(0)
    else:
        ans = input("=>only show best checkpoint file?[y/n]:")
        if ans in ['y','n','Y','N']:
            if ans in ['y','Y']:
                n_lst = [x for x in lst if 'best' in x]
                lst = n_lst
            else:
                pass
        else:
            print("=> input format error!!!!")
            sys.exit(0)
        opt_dict = {}
        for i in range(len(lst)):
            opt_dict[str(i)] = lst[i]
        # import collections
        # opt_dict = collections.OrderedDict(sorted(opt_dict.items()))
        opts = sorted(opt_dict.items(), key=lambda d: int(d[0]))
        pp.pprint(opts)
        num = input("=>please input number to show file info:")
        if str(num) in list(opt_dict.keys()):
            checkpoint_name = opt_dict[str(num)]
            checkpoint_file = os.path.join(checkpoint_path, opt_dict[str(num)])
            if os.path.exists(checkpoint_file):
                print("=> loading checkpoint %s" % (checkpoint_file))
                config_name = checkpoint_name[:19]+"_cfg.pkl"
                config_path = os.path.join(config_path, config_name)
                if not os.path.exists(config_path):
                    print("=> %s don't exist!!!!,Please checking it."%(config_path))
                    sys.exit(0)
                with open(config_path,'rb') as f:
                    cfg_dict = pkl.load(f)

                cfg.__dict__.update(cfg_dict)
                print("=> Loaded config %s"%(config_path))
                cfg.EXP_TYPE = 'E'
                cfg.EXP_NAME = "{}_{}_{}".format( cfg.EXP_TYPE, cfg.MODEL_NAME, cfg.EXP_TAG)
                cfg.SNAPSHOT_LOG = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', cfg.EXP_NAME, 'log')
                cfg.SNAPSHOT_LOG_DEBUG = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', cfg.EXP_NAME, 'log', 'debug')
                cfg.SNAPSHOT_CHECKPOINT = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', cfg.EXP_NAME, 'checkpoint')
                cfg.SNAPSHOT_SUMMARY = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', cfg.EXP_NAME, 'summary')
                cfg.SNAPSHOT_CONFIG = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', cfg.EXP_NAME, 'config')
                _io = IO()
                _io.check_folder()
                ## check GPU info
                _io.check_GPU()

                ## load model
                _model = Model()
                net = _model.select_model('tsn')
                val_transform = net.val_transform()

                ## load data
                if not cfg.REDIS_MODE:
                    v_loader,_ = val_loader_local(val_transform)
                else:
                    v_loader,_ = val_loader_remote_dpflow(val_transform)

                ## define loss function
                if torch.cuda.is_available():
                    criterion = torch.nn.CrossEntropyLoss().cuda()
                else:
                    criterion = torch.nn.CrossEntropyLoss()

                checkpoint = torch.load(checkpoint_file)
                cfg.TRAIN.START_EPOCH = checkpoint['epoch']
                cfg.TRAIN.BEST_PREC = checkpoint['best_prec1']
                own_state = net.state_dict()
                for layer_name, param in checkpoint['model_dict'].items():
                    if 'module' in layer_name:
                        layer_name = layer_name[7:]
                    if isinstance(param, torch.nn.parameter.Parameter):
                        param = param.data
                    assert param.dim() == own_state[layer_name].dim(), \
                        '{} {} vs {}'.format(layer_name, param.dim(), own_state[layer_name].dim())
                    own_state[layer_name].copy_(param)
                print("=> start epoch %d, best_prec1 %f" % (cfg.TRAIN.START_EPOCH, cfg.TRAIN.BEST_PREC))
                print("=> loaded checkpoint epoch is %d" % (cfg.TRAIN.START_EPOCH))
        else:
            print("your number is not in range!!!")
            sys.exit(0)

    ## parallel model on GPU
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net,device_ids=[0]).cuda()
        torch.backends.cudnn.benchmark = True
    else:
        pass
    import logging
    name = os.path.join(cfg.SNAPSHOT_LOG, checkpoint_name.split(".")[0] + "_E.log")
    logger = logging.getLogger("log")
    # logger.disabled = True
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=name, mode='w')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s-%(filename)s[line:%(lineno)d]-%(module)s-%(funcName)s-%(levelname)s: %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    ## save cfg
    pp.pprint(cfg.__dict__)
    name = os.path.join(cfg.SNAPSHOT_CONFIG , checkpoint_name.split(".")[0] + "_E_cfg.txt")
    print("=>config %s will be saved" % (name))
    with open(name, "w") as f:
        f.write(pp.pformat(cfg.__dict__))

    pkl_name = os.path.join(cfg.SNAPSHOT_CHECKPOINT , checkpoint_name.split(".")[0] + "_E_error_sample.pkl")
    txt_name = os.path.join(cfg.SNAPSHOT_CHECKPOINT, checkpoint_name.split(".")[0] + "_E_statistic.txt")
    epoch_start_time = time.time()
    prec1 = validate(v_loader, net, criterion, loger=logger,pkl_name=pkl_name,txt_name=txt_name)
    epoch_end_time = time.time()
    logger.info('Epoch[{0}] Epoch_Time ({etime:.3f})'.format(0,etime=(epoch_end_time-epoch_start_time)))


def validate(val_loader, model, criterion,loger = None,pkl_name=None,txt_name=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    s = Statistic(cfg.DATASET.CLASS_NUM,pkl_name,txt_name)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (_,input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5),s=s)

            losses.update(loss.cpu().detach().numpy(), input.size(0))
            top1.update(prec1.cpu().detach().numpy(), input.size(0))
            top5.update(prec5.cpu().detach().numpy(), input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.VAL.PRINT_FREQ == 0:
            loger.info(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    loger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    s.print_info(top_avg=top1.avg)

    return top1.avg

if __name__ == '__main__':
    main()