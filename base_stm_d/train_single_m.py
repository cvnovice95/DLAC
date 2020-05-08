import sys
import os
import time
import math
import torch
import pprint as pp
import numpy as np
import argparse
from torch.nn.utils import clip_grad_norm_
import multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import json
from easydict import EasyDict as edict

##self define file
from model import Model
from config import ActivityConfig as cfg
from data import train_loader_local,val_loader_local
from utils import IO,accuracy,AverageMeter,synchronize
from summary import Summary

if cfg.USE_APEX:
    from apex import amp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    cmd_options = parser.parse_args()
    cmd_options_dict = vars(cmd_options)
    cfg.__dict__.update(cmd_options_dict)
    #### TODO: get config.json, and generate edict : __C.__dict__
    with open(cfg.config, 'r') as f:
        d = json.load(f)

    __C = edict(d)
    __C.TIMESTAMP = time.strftime("%Y_%m_%d_%H:%M:%S",
                                  time.localtime())  # To initialize timestamp when importing this file
    __C.WORKSPACE_PATH = os.getcwd()  # To get current workspace path
    __C.TRAIN.RESUME = __C.PRETRAIN_MODEL_DICT[__C.MODEL_NAME][__C.BACKBONE][
        __C.RESUME_TYPE]  # _C.SNAPSHOT_CHECKPOINT + pth name
    __C.TRAIN.PRETRAIN_MODEL = __C.PRETRAIN_MODEL_DICT[__C.MODEL_NAME][__C.BACKBONE][
        __C.PRETRAIN_TYPE]  # _C.PRETRAIN_MODEL_ZOO + C.TRAIN.PRETRAIN_MODEL
    __C.EXP_NAME = "{}_{}_{}".format(__C.EXP_TYPE, __C.MODEL_NAME, __C.EXP_TAG)
    __C.SNAPSHOT_LOG = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output', __C.EXP_NAME, 'log')
    __C.SNAPSHOT_LOG_DEBUG = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output', __C.EXP_NAME, 'log', 'debug')
    __C.SNAPSHOT_CHECKPOINT = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output', __C.EXP_NAME, 'checkpoint')
    __C.SNAPSHOT_SUMMARY = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output', __C.EXP_NAME, 'summary')
    __C.SNAPSHOT_CONFIG = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output', __C.EXP_NAME, 'config')
    __C.PRETRAIN_MODEL_ZOO = os.path.join(__C.SNAPSHOT_ROOT, 'ar_output',
                                          'pretrain_model_zoo')  # save backbone pretrain model params
    cfg.__dict__.update(__C)

    if 'T' not in cfg.EXP_TYPE:
        print("=> Your Exp Type is %s, not 'T' " % (cfg.EXP_TYPE))
        sys.exit(0)
    _io = IO()

    ## check folder info
    _io.check_folder()
    ## check GPU info
    _io.check_GPU()
    ## load model
    _model = Model()
    net = _model.select_model(cfg.MODEL_NAME)
    train_transform = net.train_transform()
    val_transform = net.val_transform()
    model_params_policy = net.get_optim_policies()

    ## load dataLoader
    t_loader,_ = train_loader_local(train_transform)
    v_loader,_ = val_loader_local(val_transform)
    print("=> TRAIN:One epoch have %d steps!!!" % (len(t_loader)))
    print("=> VAL: One epoch have %d steps!!! " % (len(v_loader)))

    ## define loss function
    if torch.cuda.is_available():
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    ## define optimizer
    for group in model_params_policy:
        print('=> group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                  group['name'], len(group['params']), group['lr_mult'], group['decay_mult']))
    optimizer = torch.optim.SGD(model_params_policy,
                                cfg.TRAIN.BASE_LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    ## use apex accerlate training
    if torch.cuda.is_available():
        if cfg.USE_APEX:
            print("=> USE APEX!!!")
            net = net.cuda()
            net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
        else:
            print("=> DONT USE APEX!!!")
    else:
        pass

    ## load checkpoint
    if cfg.USE_APEX:
        _io.load_checkpoint(net, optimizer, amp)
    else:
        _io.load_checkpoint(net, optimizer, None)

    ## parallel model on GPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
                print("=> USE DataParallel")
                net = torch.nn.DataParallel(net).cuda()
                torch.backends.cudnn.benchmark = False
        else:
            print("Just use single GPU!!!!!!!!")
            net = torch.nn.DataParallel(net).cuda()
            torch.backends.cudnn.benchmark = False
    else:
        pass

    _summary = None

    from log import loger
    pp.pprint(cfg.__dict__)
    _io.save_config(cfg.__dict__)
    if cfg.USE_SUMMARY:
        print("=> Using Tensorboard Summary!!!")
        _summary = Summary(writer_dir=cfg.SNAPSHOT_SUMMARY,suffix = cfg.TIMESTAMP)



    ## trainning process
    train_start_time = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH , cfg.TRAIN.EPOCHS):
        if cfg.TRAIN.LR_STEP_TYPE == 'step':
            adjust_learning_rate_step(optimizer, epoch, cfg.TRAIN.LR_STEPS)

        # train for one epoch
        epoch_start_time = time.time()
        train(t_loader, net, criterion, optimizer, epoch,loger=loger,_summary=_summary)
        epoch_end_time = time.time()
        loger.info('Epoch[{0}] Epoch_Time ({etime:.3f})'.format(epoch,etime=(epoch_end_time-epoch_start_time)))

        # evaluate on validation set
        if (epoch + 1) % cfg.TRAIN.EVALUATE_FREQ == 0 or epoch == cfg.TRAIN.EPOCHS - 1 :

            prec1 = validate(v_loader, net, criterion, loger=loger)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > cfg.TRAIN.BEST_PREC
            cfg.TRAIN.BEST_PREC = max(prec1, cfg.TRAIN.BEST_PREC)
            if cfg.USE_APEX:
                _io.save_checkpoint({
                        'epoch': (epoch + 1),
                        'model_dict': net.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        'amp_dict': amp.state_dict(),
                        'best_prec1': cfg.TRAIN.BEST_PREC,
                    }, is_best, (epoch + 1))
            else:
                _io.save_checkpoint({
                        'epoch': (epoch + 1),
                        'model_dict': net.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        'best_prec1': cfg.TRAIN.BEST_PREC,
                    }, is_best, (epoch + 1))
    train_end_time =time.time()
    loger.info('=> Training Time [{}] ({etime:.3f} min )'.format(epoch, etime=((train_end_time - train_start_time)/60)))




def train(train_loader, model, criterion, optimizer, epoch,loger = None,_summary=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    one_epoch_steps = len(train_loader)
    end = time.time()
    iter_loader = iter(train_loader)
    for i in range(one_epoch_steps):
        _,input,target = next(iter_loader)
    # for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cfg.TRAIN.LR_STEP_TYPE == 'sgdr':
            sgdr_learning_rate(optimizer,
                               i,
                               epoch,
                               one_epoch_steps,
                               cfg.TRAIN.PERIOD_EPOCH,
                               cfg.TRAIN.BASE_LR,
                               lr_decay=1.0,
                               warmup_epoch=cfg.TRAIN.WARMUP_EPOCH)
        if input.size(0) % cfg.TRAIN.BATCH_SIZE != 0:
            continue

        if torch.cuda.is_available():
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # output = (output,)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5),s=None)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        if cfg.USE_APEX and torch.cuda.is_available():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if cfg.TRAIN.CLIP_GRADIENT is not None:
            total_norm = clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRADIENT)
            # if total_norm > cfg.TRAIN.CLIP_GRADIENT:
            #     print("clipping gradient: {} with coef {}".format(total_norm,cfg.TRAIN.CLIP_GRADIENT / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        print_train_log(epoch,i,one_epoch_steps,batch_time,data_time,losses,top1,top5,
                        optimizer.param_groups[-1]['lr'],loger=loger)
        if _summary is not None:
            _summary.add_train_scalar(i, epoch, one_epoch_steps,
                                          losses.avg, top1.avg, top5.avg,
                                          losses.avg, top1.avg, top5.avg,
                                          optimizer.param_groups[-1]['lr'])
        end = time.time()

def validate(val_loader, model, criterion,loger = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            prec1, prec5 = accuracy(output.data, target, topk=(1,5),s=None)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


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

    return top1.avg

def print_train_log(epoch,i,one_epoch_steps,
                    batch_time,data_time,losses,
                    top1,top5,lr,loger = None):
    if i % cfg.TRAIN.PRINT_FREQ == 0:
        loger.info(('=> Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                         epoch, i, one_epoch_steps,lr=lr,
                         batch_time=batch_time,
                         data_time=data_time,
                         loss=losses,
                         top1=top1, top5=top5)))

def sgdr_learning_rate(optimizer,batch_id,epoch,one_epoch_steps,period_epoch,base_lr,lr_decay=1.0,warmup_epoch=None):

    total_iteration_idx = batch_id+epoch*one_epoch_steps
    restart_period = period_epoch * one_epoch_steps
    warmup_steps = 0 if warmup_epoch is None else int(one_epoch_steps * warmup_epoch)
    if warmup_epoch and total_iteration_idx < warmup_steps:
        lr =base_lr * total_iteration_idx / warmup_steps
    else:
        cnt = (total_iteration_idx - warmup_steps) // restart_period
        batch_idx = (total_iteration_idx - warmup_steps) % restart_period

        radians = math.pi * (batch_idx / restart_period)
        lr = 0.5 * (1.0 + math.cos(radians)) * (lr_decay ** cnt) * base_lr
    ## update lr
    new_lr = lr
    decay = cfg.TRAIN.WEIGHT_DECAY
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def adjust_learning_rate_step(optimizer, epoch, lr_steps):

    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = cfg.TRAIN.BASE_LR * decay
    decay = cfg.TRAIN.WEIGHT_DECAY
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if __name__ == '__main__':
    main()