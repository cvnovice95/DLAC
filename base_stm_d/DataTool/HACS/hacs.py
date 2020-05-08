# import urllib.request
# import rarfile           ## sudo apt install unrar | pip3 install rarfile
import sys
from multiprocessing import Pool
import cv2
import os
import json
from functools import partial
import pprint as pp
'''
date:2019-12-09 14:52:12
author: jinyuanfeng
description: None
'''

'''
Dataset Structure:
Format:
/ROOT_PATH/dataset_name/class_name/video_name.mp4  save original dataset
/ROOT_PATH/dataset_name/class_name/video_name/     save separating frame dataset
'''

'''
Label File Structure:
Format:
class_name/video_name.mp4 label                                     original dataset's label file
/ROOT_PATH/dataset_name/class_name/video_name/ frame_num label(int) separating frame dataset's label file
'''

train_label_org_path = '/mnt/lustre/fengjinyuan/data/datasets/HACS_label/hacs_segments_untrimmed_train_org.txt'
val_label_org_path = '/mnt/lustre/fengjinyuan/data/datasets/HACS_label/hacs_segments_untrimmed_val_org.txt'
_video_seq_path = '/mnt/lustre/fengjinyuan/data/datasets/HACSSegmentsSeqNew/'
_video_org_path = '/mnt/lustre/fengjinyuan/data/datasets/HACSSegments/'

## HACS JSON prasing code: /mnt/lustre/fengjinyuan/codebase/jupyter_code_kit/preprocess_HACS.ipynb

def video2frame(video_org_path,video_seq_path,file_info):
    print("=>",video_seq_path)
    '''
    video_org_path: format: video_org_dir/
                    structure: video_org_dir/class_dir/video_name.avi['mp4']
                    or structure: video_org_dir/video_name.avi['mp4']
    video_seq_path: format: video_seq_dir/
                    structure: video_seq_dir/class_dir/video_name_dir/
    file_name:  format:  class_dir/video_name.avi['mp4']
    info.txt: format: frame number,frame rate, duration,height,width
    '''
    if isinstance(file_info,str):
        file_name = file_info # class_dir/video_name.avi['mp4']
        file_name_path = file_info.split(".")[0] # class_dir/video_name/
    elif isinstance(file_info,list):
        file_name = file_info[0] # video_name.avi['mp4']
        file_name_path = os.path.join(file_info[1],file_info[0].split(".")[0]) # class_dir/video_name/
    else:
        print("=> file_info type don't support!!!,type:",type(file_info))
        sys.exit(0)

    print("=> %s will be processed"%(file_name))
    file_path = os.path.join(video_org_path,file_name)
    save_frame_path = os.path.join(video_seq_path,file_name_path)
    if os.path.exists(save_frame_path):
        print("=> %s has existed,please delete it, try again"%(save_frame_path))
        sys.exit(0)
    else:
        os.makedirs(save_frame_path)
    if not os.path.exists(file_path):
        print("=> %s don't exist,please checking it"%(file_path))
        sys.exit(0)
    cap = cv2.VideoCapture(file_path)
    print("=> frame hight:",cap.get(4))
    print("=> frame width",cap.get(3))
    print("=> frame number:",cap.get(7))
    print("=> frame rate:",cap.get(5))
    duration = float(cap.get(7)/cap.get(5))
    print("=> duration:",duration)
    cnt = 0
    for i in range(int(cap.get(7))):
        success,frame = cap.read()
        if success:
            cnt += 1
            save_frame_name = os.path.join(save_frame_path,"img_{:05d}.jpg".format(i))
            cv2.imwrite(save_frame_name,frame)
    print("=> video to frame done!!!")
    with open(os.path.join(save_frame_path,"info.csv"),'w') as f:
        f.write("%d %f %f %d %d"%(cnt,cap.get(5),duration,cap.get(4),cap.get(3)))

def video2frame_pool(video_org_path,video_seq_path,meta_file_path):
    #TODO

    # meta_file_list = [[x.strip().split(',')[0],x.strip().split(',')[2]] for x in open(os.path.join(meta_file_path),'r')]
    meta_file_list = [x.strip().split(' ')[0] for x in open(os.path.join(meta_file_path), 'r')]
    pool = Pool(10)
    p_video2frame = partial(video2frame,video_org_path,video_seq_path)
    pool.map_async(p_video2frame,meta_file_list)
    pool.close()
    pool.join()
    print("=> done!!!")

def somethingv2_process_seq_metafile(metafile_org_path,
                                video_seq_path,
                                metafile_seq_path,
                                save_name=None):
    _path = metafile_org_path
    if not os.path.exists(_path):
        print("=> %s don't exist,please checking it" % (_path))
        sys.exit(0)
    lst = []
    cnt = 0
    for x in open(_path,'r'):
        cnt += 1
        print("start process %d"%(cnt))
        item = x.strip().split(',')
        video_name = os.path.join(item[2],item[0].strip().split('.')[0])
        label = item[1]
        _info_path = os.path.join(video_seq_path,video_name,"info.csv")
        if not os.path.exists(_path):
            print("=> %s don't exist,please checking it" % (_info_path))
            sys.exit(0)
        with open(_info_path,"r") as f:
            data = f.readline()
            seq_len = data.strip().split(' ')[0]
        lst.append((video_name,seq_len,label))
    print("=> all len is %d"%(len(lst)))
    with open(os.path.join(metafile_seq_path,save_name),"w") as f:
        for x in lst:
            f.write("%s,%s,%s\n"%(x[0],x[1],x[2]))
    print("=> done!")



if __name__ == '__main__':
    # create_dataset_dir()
    # somethingv2_process_metafile()
    # ## video to frame sequence
    video2frame_pool(_video_org_path, _video_seq_path, train_label_org_path)
    video2frame_pool(_video_org_path, _video_seq_path, val_label_org_path)
    ## generate frame sequence corresponding label
    # video_seq_path = os.path.join(ROOT_PATH,dataset_name,video_seq)
    # metafile_seq_path = os.path.join(ROOT_PATH,dataset_name,metafile_seq)
    # for x in [(video2frame_train_metafile,'train_split1.csv'),(video2frame_val_metafile,'val_split1.csv')]:
    #     somethingv2_process_seq_metafile(x[0],
    #                                 video_seq_path,
    #                                 metafile_seq_path,
    #                                 save_name=x[1])
    print("Congratulations!!!")



