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

##self define file
from model import Model
from config import ActivityConfig as cfg
from data import train_loader_local,val_loader_local
from utils import IO,accuracy,AverageMeter,synchronize
from summary import Summary


if cfg.USE_APEX:
    from apex import amp
if cfg.REDIS_MODE:
    from data import train_loader_remote_dpflow, val_loader_remote_dpflow

def main():
    if 'T' not in cfg.EXP_TYPE:
        print("=> Your Exp Type is %s, not 'T' " % (cfg.EXP_TYPE))
        sys.exit(0)
    _io = IO()
    ## config distributed args
    if cfg.USE_DISTRIBUTED:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)  # local_rank from torch.distributed.launch
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--dist_backend', type=str, default='nccl')
        cmd_options = parser.parse_args()
        cmd_options_dict = vars(cmd_options)
        cfg.__dict__.update(cmd_options_dict)
        dist_cfg = {}
        dist_cfg['world_size'] = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        dist_cfg['rank'] = int(os.environ["RANK"]) if "RANK" in os.environ else 0
        dist_cfg['master_port'] = os.environ["MASTER_PORT"] if "MASTER_PORT" in os.environ else 12345
        cfg.__dict__.update(dist_cfg)

        ## init distribute config
        multiprocessing.set_start_method('forkserver')
        torch.cuda.set_device(cfg.local_rank)
        master_addr = None

        if cfg.num_nodes == 1:
            master_addr = '127.0.0.1'
        else:
            pass
        # init_method is tcp type because master ip is available after get_local_ip runs
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method="tcp://{}:{}".format(master_addr, cfg.master_port),
                                rank=cfg.rank,
                                world_size=cfg.world_size)
        synchronize()

        print("=> world_size: {},rank: {}, master_addr: {}, master_port: {},local_rank: {}".format( \
            cfg.world_size, cfg.rank, master_addr, cfg.master_port, cfg.local_rank))
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
    if cfg.USE_SYNC_BN:
        print("=> Use Sync BN!!!")
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    model_params_policy = net.get_optim_policies()

    ## load dataLoader
    if not cfg.REDIS_MODE:
        if cfg.USE_DISTRIBUTED:
            t_loader,_,t_sampler = train_loader_local(train_transform)
            v_loader,_ = val_loader_local(val_transform)
        else:
            t_loader,_ = train_loader_local(train_transform)
            v_loader,_ = val_loader_local(val_transform)
    else:
        t_loader,_ = train_loader_remote_dpflow(train_transform)
        v_loader,_ = val_loader_remote_dpflow(val_transform)
    print("=> One epoch have %d steps!!!"%(len(t_loader)))

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
            if cfg.USE_DISTRIBUTED:
                    net = net.cuda()
                    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
                    # NO model adaption allowed after DistributedDataParallel wrap
                    net = DistributedDataParallel(net, device_ids=[cfg.local_rank], output_device=cfg.local_rank,find_unused_parameters=True)
                    torch.backends.cudnn.benchmark = False
            else:
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
    for epoch in range(cfg.TRAIN.START_EPOCH , cfg.TRAIN.EPOCHS):
        if cfg.TRAIN.LR_STEP_TYPE == 'step':
            adjust_learning_rate_step(optimizer, epoch, cfg.TRAIN.LR_STEPS)

        # train for one epoch
        epoch_start_time = time.time()
        if cfg.USE_DISTRIBUTED:
            t_sampler.set_epoch(epoch)
        train(t_loader, net, criterion, optimizer, epoch,loger=loger,_summary=_summary)
        epoch_end_time = time.time()
        loger.info('Epoch[{0}] Epoch_Time ({etime:.3f})'.format(epoch,etime=(epoch_end_time-epoch_start_time)))

        # evaluate on validation set
        if (epoch + 1) % cfg.TRAIN.EVALUATE_FREQ == 0 or epoch == cfg.TRAIN.EPOCHS - 1 :
            if cfg.USE_DISTRIBUTED:
                if cfg.rank == 0:
                    prec1 = validate(v_loader, net, criterion,loger=loger)
                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > cfg.TRAIN.BEST_PREC
                    cfg.TRAIN.BEST_PREC = max(prec1, cfg.TRAIN.BEST_PREC)
                    if cfg.USE_APEX:
                        _io.save_checkpoint({
                            'epoch': (epoch + 1),
                            'model_dict': net.state_dict(),
                            'optimizer_dict':optimizer.state_dict(),
                            'amp_dict':amp.state_dict(),
                            'best_prec1': cfg.TRAIN.BEST_PREC,
                        }, is_best,(epoch+1))
                    else:
                        _io.save_checkpoint({
                            'epoch': (epoch + 1),
                            'model_dict': net.state_dict(),
                            'optimizer_dict': optimizer.state_dict(),
                            'best_prec1': cfg.TRAIN.BEST_PREC,
                        }, is_best, (epoch + 1))
            else:
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
    loger.info('=> Training Time ({etime:.3f} min )'.format(epoch, etime=((train_end_time - train_start_time)/60)))




def train(train_loader, model, criterion, optimizer, epoch,loger = None,_summary=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global_losses = AverageMeter()
    global_top1 = AverageMeter()
    global_top5 = AverageMeter()

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


        collect_global_info(losses,top1,top5,global_losses,global_top1,global_top5,input.size(0))
        print_train_log(epoch,i,one_epoch_steps,batch_time,data_time,losses,top1,top5,
                        optimizer.param_groups[-1]['lr'],global_top1,global_top5,global_losses,loger=loger)
        if _summary is not None:
            if cfg.USE_DISTRIBUTED:
                if cfg.rank == 0:
                    _summary.add_train_scalar(i, epoch, one_epoch_steps,
                                              losses.avg, top1.avg, top5.avg,
                                              global_losses.avg, global_top1.avg,global_top5.avg ,
                                              optimizer.param_groups[-1]['lr'])
            else:
                _summary.add_train_scalar(i, epoch, one_epoch_steps,
                                          losses.avg, top1.avg, top5.avg,
                                          losses.avg, top1.avg, top5.avg,
                                          optimizer.param_groups[-1]['lr'])
        end = time.time()


def collect_global_info(losses,top1,top5,global_losses,global_top1,global_top5,batch_size):
        """
        Collect global loss and prec using dist.all_reduce
        """
        if cfg.USE_DISTRIBUTED:
            world_size = dist.get_world_size()
        else:
            global_top1 = top1
            global_top5 = top5
            global_losses = losses
            return

        prec1 = torch.as_tensor(top1.val) / world_size
        prec5 = torch.as_tensor(top5.val) / world_size
        loss = torch.as_tensor(losses.val) / world_size
        global_batch = torch.as_tensor(batch_size)
        if torch.cuda.is_available():
            prec1, prec5, loss, global_batch = prec1.cuda(), prec5.cuda(), loss.cuda(), global_batch.cuda()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(prec1, op=dist.ReduceOp.SUM)
        dist.all_reduce(prec5, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_batch, op=dist.ReduceOp.SUM)
        global_top1.update(prec1.item(), global_batch.item())
        global_top5.update(prec5.item(), global_batch.item())
        global_losses.update(loss.item())
        synchronize()

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
def print_train_log(epoch,
                    i,
                    one_epoch_steps,
                    batch_time,
                    data_time,
                    losses,
                    top1,
                    top5,
                    lr,
                    global_top1,
                    global_top5,
                    global_losses,
                    loger = None):
    if cfg.USE_DISTRIBUTED:
        if i % cfg.TRAIN.PRINT_FREQ == 0 and cfg.rank == 0:
            loger.info(('Rank[{0:0>2d}] => Epoch: [{1}][{2}/{3}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                        'G_Loss {g_losses.val:.4f} ({g_losses.avg:.4f})\t'
                        'G_Prec@1 {g_top1.val:.3f} ({g_top1.avg:.3f})\t'
                        'G_Prec@5 {g_top5.val:.3f} ({g_top5.avg:.3f})\t'.format(
                        cfg.rank, epoch, i, one_epoch_steps, lr=lr,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1, top5=top5,
                        g_losses=global_losses,
                        g_top1=global_top1,g_top5=global_top5)))
    else:
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