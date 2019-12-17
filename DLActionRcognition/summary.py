import numpy as np
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

# TODO: replace by pytorch official tensorboard


class Summary(object):
    def __init__(self, writer_dir=None, args=None,suffix = None):
        if writer_dir:
            self.writer = SummaryWriter(writer_dir,filename_suffix=suffix)
        else:
            self.writer = SummaryWriter()
        self.args = args

    # kwargs = optimizer, loss, prec1, prec5, i,
    def add_train_scalar(self, i, epoch, epoch_len,losses,top1,top5,glosses,gtop1,gtop5,lr):
        # summary training lr, loss, accuracy
        # TODO: add weight_decay loss
        total_step = i + epoch * epoch_len

        self.writer.add_scalar('train/lr', lr, total_step)
        self.writer.add_scalar('train/prec1', top1/100, total_step)
        self.writer.add_scalar('train/prec5', top5/100, total_step)
        self.writer.add_scalar('train/g_prec1', gtop1 / 100, total_step)
        self.writer.add_scalar('train/g_prec5', gtop5 / 100, total_step)
        self.writer.add_scalar('train/loss', losses, total_step)
        self.writer.add_scalar('train/g_loss', glosses, total_step)


    # kwargs = loss, prec1, prec5, etc.
    def add_valid_scalar(self, epoch, epoch_len, **kwargs):
        total_step = epoch * epoch_len
        for key in kwargs:
            val = kwargs[key]
            if key.startswith('prec'):
                val /= 100
            self.writer.add_scalar('valid/'+key, val, total_step)

    def add_histogram(self, root_name, model, i, epoch, epoch_len, grad=False):
        total_step = i + epoch * epoch_len
        if isinstance(model, list):
            model = np.stack(model, axis=0)
        if isinstance(model, torch.Tensor):
            model = model.clone().cpu().data.numpy()
        if isinstance(model, np.ndarray):
            self.writer.add_histogram(root_name + '/', model, total_step)
            return

        for name, param in model.named_parameters():
            self.writer.add_histogram(root_name + '/' + name, param.clone().cpu().data.numpy(), total_step)
            if grad:
                self.writer.add_histogram('grad/' + name, param.grad.clone().cpu().data.numpy(), total_step)

    def add_graph(self, model, input_size):
        # if i == 0 and epoch == self.args.start_epoch:
        #     self.writer.add_graph(model, input_var)
        demo_input = torch.rand(input_size)
        self.writer.add_graph(model, demo_input)

    def add_image(self, name, frames, i, epoch, epoch_len):
        total_step = i + epoch * epoch_len

        x = frames.clone().cpu().data.numpy()
        self.writer.add_histogram(name+'histogram', x, total_step)
        x = x.transpose(0, 2, 1, 3, 4)
        x = np.ascontiguousarray(x, dtype=np.float32)
        x = x.reshape(-1, *x.shape[-3:])

        grid = vutils.make_grid(torch.from_numpy(x), normalize=True)
        self.writer.add_image(name+'image', grid, total_step)

    def close(self):
        self.writer.close()

