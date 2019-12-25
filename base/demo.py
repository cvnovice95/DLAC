import numpy as np
from PIL import Image
import torch
import imutils
from torch.autograd import Variable
from torch.nn import functional as F
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import cv2
import os
import sys
import pprint as pp
## self define file
from model import Model
from transform import *
from config import ActivityConfig as cfg

def load_frames(frames, num_frames=5):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))

def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,(1, int(height / 8)),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames

def render_single_frame(frame, label):
    cv2.putText(frame, "class label: " + label,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
    return frame


def render_video(net,transform,cls_dict,fps,reader,writer):
    pred = "";
    buffer_frame = []
    cnt = 0
    while True:

        frame= reader.read()
        # print(frame.shape)
        if not reader.more():
            break
        frame = render_single_frame(frame, pred)

        frame = cv2.resize(frame, (640, 480))
        cnt += 1
        writer.write(frame)
        # cv2.imwrite("./output/"+str(cnt)+".jpg", frame)


        input_pill = Image.fromarray(frame)

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        if (len(buffer_frame) < 10):
            buffer_frame.append(input_pill)
        else:
            input_frames = load_frames(buffer_frame)
            data = transform(input_frames)
            input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)

            with torch.no_grad():
                logits = net(input)
                h_x = torch.mean(F.softmax(logits, 1), dim=0).data
                probs, idx = h_x.sort(0, True)
                # print(type(idx))
                # print(idx)
                pred = cls_dict[str(idx.cpu().detach().numpy()[0])]
                buffer_frame[:-1] = buffer_frame[1:]
                buffer_frame[-1] = input_pill
        # update the FPS counter
        fps.update()
    fps.stop()
    print("=>elasped time: {:.2f}".format(fps.elapsed()))
    print("=>approx. FPS: {:.2f}".format(fps.fps()))
    reader.stop()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--cls_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    args = parser.parse_args()
    ## checking video_path,model_path,output_path
    if not os.path.exists(args.video_path):
        print("=> %s don't exist!!!,Please checking it."%(args.video_path))
        sys.exit(0)
    if not os.path.exists(args.model_path):
        print("=> %s don't exist!!!,Please checking it." % (args.model_path))
        sys.exit(0)
    if not os.path.exists(args.output_path):
        print("=> %s don't exist!!!,Please checking it." % (args.output_path))
        sys.exit(0)
    if not os.path.exists(args.cls_path):
        print("=> %s don't exist!!!,Please checking it." % (args.cls_path))
        sys.exit(0)

    ## prepare model
    _model = Model()
    cfg.MODEL_NAME = 'tsn'
    cfg.BACKBONE = 'resnet50'
    cfg.PRETRAIN_TYPE = 'imagenet'
    cfg.TRAIN.PRETRAIN_MODEL = cfg.PRETRAIN_MODEL_DICT[cfg.MODEL_NAME][cfg.BACKBONE][cfg.PRETRAIN_TYPE]
    cfg.DATASET.CLASS_NUM = 51
    cfg.TRAIN.MODALITY = 'RGB'

    net = _model.select_model('tsn')
    val_transform = net.val_transform()
    ## load model
    checkpoint = torch.load(args.model_path)
    own_state = net.state_dict()
    for layer_name, param in checkpoint['model_dict'].items():
        if 'module' in layer_name:
            layer_name = layer_name[7:]
        if isinstance(param, torch.nn.parameter.Parameter):
            param = param.data
        assert param.dim() == own_state[layer_name].dim(), \
            '{} {} vs {}'.format(layer_name, param.dim(), own_state[layer_name].dim())
        own_state[layer_name].copy_(param)
    print("=> start epoch %d, best_prec1 %f" % (checkpoint['epoch'], checkpoint['best_prec1']))
    print("=> loaded checkpoint epoch is %d" % (checkpoint['epoch']))
    ## parse cls as dict
    cls={}
    with open(args.cls_path,"r") as f:
        for x in f:
            item = x.strip().split(',')
            cls[item[0]]=item[1]
    pp.pprint(cls)
    print("=> sampling frame from video by Sample Thread.")
    vs = FileVideoStream(args.video_path).start()
    fps = FPS().start()
    writer = cv2.VideoWriter(os.path.join(args.output_path,'output.avi'), cv2.VideoWriter_fourcc('D','I','V','X'),25.0,(640, 480))
    net.cuda().eval()
    render_video(net, val_transform, cls,fps,vs,writer)


if __name__ == '__main__':
   main()

