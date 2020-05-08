import sys
import os
import time
import math
import torch
import pprint as pp
import numpy as np
import argparse
from torch.nn.utils import clip_grad_norm_
import multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
## sensetime
import linklink as link

##self define file
from model import Model
from config import ActivityConfig as cfg
from data import train_loader_local_st,val_loader_local_st
from utils import IO,accuracy,AverageMeter,dist_init, reduce_gradients, DistModule
from summary import Summary

mp.set_start_method('spawn', force=True)
_io = IO()
def main():
    torch.cuda.empty_cache()
    if 'T' not in cfg.EXP_TYPE:
        print("=> Your Exp Type is %s, not 'T' " % (cfg.EXP_TYPE))
        sys.exit(0)

    ## config distributed args
    if cfg.USE_DISTRIBUTED:
        rank, world_size = dist_init()
        dist_cfg = {}
        dist_cfg['world_size'] = world_size
        dist_cfg['rank'] = rank
        cfg.__dict__.update(dist_cfg)
        print("=> world_size: {},rank: {}".format( cfg.world_size, cfg.rank))
        if cfg.rank == 0:
            _io.check_folder()
            ## check GPU info
            _io.check_GPU()
    else:
        _io.check_folder()
        ## check GPU info
        _io.check_GPU()

    ## load model
    _model = Model()
    net = _model.select_model(cfg.MODEL_NAME)
    train_transform = net.train_transform()
    val_transform = net.val_transform()
    model_params_policy = net.get_optim_policies()
    ## sensetime
    net = net.cuda()
    net = DistModule(net, True)

    ## load dataLoader
    if cfg.USE_DISTRIBUTED:
        t_loader,_,t_sampler = train_loader_local_st(train_transform)
        v_loader,_,v_sampler = val_loader_local_st(val_transform)
    else:
        t_loader,_ = train_loader_local(train_transform)
        v_loader,_ = val_loader_local(val_transform)
    print("=> TRAIN:One epoch have %d steps!!!"%(len(t_loader)))
    print("=> VAL: One epoch have %d steps!!! "%(len(v_loader)))

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
    ## load checkpoint
    _io.load_checkpoint(net, optimizer, None)

    ## parallel model on GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    else:
        pass
    _summary = None
    if cfg.USE_DISTRIBUTED:
        from log import loger
        if cfg.rank == 0:
            pp.pprint(cfg.__dict__)
            _io.save_config(cfg.__dict__)
            if cfg.USE_SUMMARY:
                print("=> Using Tensorboard Summary!!!")
                _summary = Summary(writer_dir=cfg.SNAPSHOT_SUMMARY,suffix = cfg.TIMESTAMP)
    else:
        from log import loger
        pp.pprint(cfg.__dict__)
        _io.save_config(cfg.__dict__)
        if cfg.USE_SUMMARY:
            print("=> Using Tensorboard Summary!!!")
            _summary = Summary(writer_dir=cfg.SNAPSHOT_SUMMARY,suffix = cfg.TIMESTAMP)

    ## trainning process
    train_start_time = time.time()
    torch.cuda.memory_allocated(0)
    train(t_loader, v_loader,net, criterion, optimizer, cfg.TRAIN.START_EPOCH,loger=loger,_summary=_summary)
    link.finalize()
    train_end_time =time.time()
    loger.info('=> Training Time ({etime:.3f} min )'.format(etime=((train_end_time - train_start_time)/60)))

def train(train_loader,val_loader, model, criterion, optimizer, epoch,loger = None,_summary=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    world_size = link.get_world_size()
    rank = link.get_rank()

    one_epoch_steps = len(train_loader)
    end = time.time()
    iter_loader = iter(train_loader)

    for i in range(epoch,one_epoch_steps):
        # i = epoch
        _,input,target = next(iter_loader)
    # for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cfg.TRAIN.LR_STEP_TYPE == 'sgdr':
            sgdr_learning_rate_iter(optimizer,
                                    i,
                                    cfg.TRAIN.PERIOD_ITERATION,
                                    cfg.TRAIN.BASE_LR,
                                    lr_decay=1.0,
                                    warmup_iteration=cfg.TRAIN.WARMUP_ITERATION)
        if cfg.TRAIN.LR_STEP_TYPE == 'step':
            adjust_learning_rate_setp_iter(optimizer, i, cfg.TRAIN.LR_STEPS_ITERATION)

        # if input.size(0) % cfg.TRAIN.BATCH_SIZE != 0:
        #     continue

        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # output = (output,)
        loss = criterion(output, target_var)/world_size
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1,5),s=None)

        reduced_loss = loss.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size

        global_batch = torch.as_tensor(input.size(0))

        link.allreduce(reduced_loss)
        link.allreduce(reduced_prec1)
        link.allreduce(reduced_prec5)
        link.allreduce(global_batch)

        losses.update(reduced_loss.item(),  global_batch.item())
        top1.update(reduced_prec1.item(),   global_batch.item())
        top5.update(reduced_prec5.item(),   global_batch.item())


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if cfg.TRAIN.CLIP_GRADIENT is not None:
            total_norm = clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRADIENT)
            # if total_norm > cfg.TRAIN.CLIP_GRADIENT:
            #     print("clipping gradient: {} with coef {}".format(total_norm,cfg.TRAIN.CLIP_GRADIENT / total_norm))
        reduce_gradients(model, True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)



        print_train_log(i,input.size(0),global_batch.item(),rank,one_epoch_steps,batch_time,data_time,losses,top1,top5,
                        optimizer.param_groups[-1]['lr'],loger=loger)
        # evaluate on validation set
        torch.cuda.memory_allocated(0)
        if (i + 1) % cfg.TRAIN.EVALUATE_FREQ == 0:
            prec1 = validate((i+1),val_loader, model, criterion,optimizer, loger=loger)

        # if _summary is not None:
        #     if cfg.USE_DISTRIBUTED:
        #         if cfg.rank == 0:
        #             _summary.add_train_scalar(i, epoch, one_epoch_steps,
        #                                       losses.avg, top1.avg, top5.avg,
        #                                       global_losses.avg, global_top1.avg,global_top5.avg ,
        #                                       optimizer.param_groups[-1]['lr'])
        #     else:
        #         _summary.add_train_scalar(i, epoch, one_epoch_steps,
        #                                   losses.avg, top1.avg, top5.avg,
        #                                   losses.avg, top1.avg, top5.avg,
        #                                   optimizer.param_groups[-1]['lr'])
        end = time.time()

def validate(batch_id,val_loader, model, criterion,optimizer,loger = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    rank = link.get_rank()
    world_size = link.get_world_size()

    end = time.time()
    with torch.no_grad():
        for i, (_,input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var) /world_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1,5),s=None)

            # reduced_loss = loss.clone()
            # reduced_prec1 = prec1.clone() / world_size
            # reduced_prec5 = prec5.clone() / world_size
            # link.allreduce(reduced_loss)
            # link.allreduce(reduced_prec1)
            # link.allreduce(reduced_prec5)
            #
            # losses.update(reduced_loss.item(), input.size(0))
            # top1.update(reduced_prec1.item(), input.size(0))
            # top5.update(reduced_prec5.item(), input.size(0))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.VAL.PRINT_FREQ == 0 and rank == 0:
                loger.info(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print("Rank[{0}] start allreduce.....".format(rank))
    print("Rank[{0}] Data Size[{1}]".format(rank,losses.count))
    total_num = torch.Tensor([losses.count])
    loss_sum = torch.Tensor([losses.avg * losses.count])
    top1_sum = torch.Tensor([top1.avg * top1.count])
    top5_sum = torch.Tensor([top5.avg * top5.count])
    link.allreduce(total_num)
    link.allreduce(loss_sum)
    link.allreduce(top1_sum)
    link.allreduce(top5_sum)
    final_loss = loss_sum.item() / total_num.item()
    final_top1 = top1_sum.item() / total_num.item()
    final_top5 = top5_sum.item() / total_num.item()
    print("Rank[{0}] end allreduce".format(rank))
    if rank == 0:

        loger.info(('Testing Results: BestPrec@1 {:.3f} Prec@1 {:.3f} Prec@5 {:.3f} Loss {:.5f}'.format(cfg.TRAIN.BEST_PREC,final_top1, final_top5, final_loss)))
        # loger.info(('Testing Results: Prec@1 {:.3f} Prec@5 {:.3f} Loss {:.5f}'.format(top1.avg, top5.avg, losses.avg)))
        # remember best prec@1 and save checkpoint
        is_best = final_top1 > cfg.TRAIN.BEST_PREC
        cfg.TRAIN.BEST_PREC = max(final_top1, cfg.TRAIN.BEST_PREC)
        _io.save_checkpoint({
            'epoch': batch_id,
            'model_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'best_prec1': cfg.TRAIN.BEST_PREC,
        }, is_best, batch_id)
    model.train()
    return final_top1

def print_train_log(i,bs,gbs,rank,one_epoch_steps,batch_time,data_time, \
                    losses,top1,top5,lr,loger = None):
    if i % cfg.TRAIN.PRINT_FREQ == 0 and rank == 0:
            loger.info(('Rank[{0:0>2d}] => Iter: [{1}/{2}], BS:[{3}] GBS:[{4}] lr: {lr:.7f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                        rank, i, one_epoch_steps,bs,gbs,lr=lr,
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

def sgdr_learning_rate_iter(optimizer, batch_id, period_iteration, base_lr, lr_decay=1.0,warmup_iteration=None):

        total_iteration_idx = batch_id
        restart_period = period_iteration
        warmup_steps = 0 if warmup_iteration is None else int(warmup_iteration)
        if warmup_iteration and total_iteration_idx < warmup_steps:
            lr = base_lr * total_iteration_idx / warmup_steps
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

def adjust_learning_rate_setp_iter(optimizer, batch_id, lr_steps):

    decay = 0.1 ** (sum(batch_id>= np.array(lr_steps)))
    lr = cfg.TRAIN.BASE_LR * decay
    decay = cfg.TRAIN.WEIGHT_DECAY
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

if __name__ == '__main__':
    main()
