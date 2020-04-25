import urllib.request
import rarfile           ## sudo apt install unrar | pip3 install rarfile
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
/ROOT_PATH/dataset_name/
                    |-MetafileOrg/
                    |-MetafileSeq/
                            |-train.csv
                            |-val.csv
                    |-VideoOrg/
                            |-dataset_name/class_name/xxx.avi (or dataset_name/xxx.avi)
                    |-VideoSeq/
                            |-class_name/xxx.avi
'''
'''
train.csv/val.csv
Format:
/class_name/video_name.avi,the number of frame,label(int)
'''

ROOT_PATH = "/data"       ##TODO: You only modify this flag according to your ROOT_PATH
dataset_name = 'SomethingV2'
metafile_org = 'MetafileOrg'
metafile_seq = 'MetafileSeq'
video_org = 'VideoOrg'
video_seq = 'VideoSeq'
metafile_org_list = [
    'something-something-v2-labels.json',
    'something-something-v2-test.json',
    'something-something-v2-train.json',
    'something-something-v2-validation.json'
]
video2frame_train_metafile = os.path.join(ROOT_PATH,dataset_name,metafile_org,'train_split1.csv')
video2frame_val_metafile = os.path.join(ROOT_PATH,dataset_name,metafile_org,'val_split1.csv')
video2frame_video_org_path = os.path.join(ROOT_PATH,dataset_name,video_org,'somethingV2')
video2frame_video_seq_path = os.path.join(ROOT_PATH,dataset_name,video_seq)

def create_dataset_dir():
    for x in [metafile_org,metafile_seq,video_org,video_seq]:
        path = os.path.join(ROOT_PATH, dataset_name, x)
        if os.path.isdir(path):
            print("=> %s has existed"%(path))
        else:
            print("=> %s will be created!"%(path))
            os.makedirs(path)
    print("=> create dir done!!!")

def somethingv2_process_metafile():
    # checking file
    for x in metafile_org_list:
        _path = os.path.join(ROOT_PATH,dataset_name,metafile_org,x)
        if not os.path.exists(_path):
            print("=> %s don't exists,please checking it.Put it into %s"%(_path,os.path.join(ROOT_PATH,dataset_name,metafile_org)))
            sys.exit(0)
    # generate cls_split1.csv
    with open(os.path.join(ROOT_PATH,dataset_name,metafile_org,metafile_org_list[0]),'r') as f:
        label_json = json.load(f)
    sorted_keys = sorted(list(label_json.keys()))
    cls_dict = {}
    with open(os.path.join(ROOT_PATH,dataset_name,metafile_org,'cls_split1.csv'),'w') as f:
        for x in sorted_keys:
            f.write("%s,%s\n"%(label_json[x],x.replace(' ','_').replace(',','_')))
            cls_dict[x.replace(' ','_').replace(',','_')] = label_json[x]
    print("=> cls_split1.csv save at %s"%(os.path.join(ROOT_PATH,dataset_name,metafile_org)))
    print("=> cls_split1.csv len %d"%(len(cls_dict.keys())))
    # generate val_split1.csv
    with open(os.path.join(ROOT_PATH,dataset_name,metafile_org,metafile_org_list[3]),'r') as f:
        val_json = json.load(f)
    print("=> val_json len %d"%(len(val_json)))
    with open(os.path.join(ROOT_PATH, dataset_name, metafile_org, 'val_split1.csv'), 'w') as f:
        for i in range(len(val_json)):
            label = val_json[i]['template'].replace('[', '').replace(']', '').replace(' ', '_').replace(',', '_')
            f.write("%s.webm,%s,%s\n"%(val_json[i]['id'],cls_dict[label],label))
    print("=> val_split1.csv save at %s" % (os.path.join(ROOT_PATH, dataset_name, metafile_org)))
    # generate train_split1.csv
    with open(os.path.join(ROOT_PATH, dataset_name, metafile_org, metafile_org_list[2]), 'r') as f:
        train_json = json.load(f)
    print("=> train_json len %d" % (len(train_json)))
    with open(os.path.join(ROOT_PATH, dataset_name, metafile_org, 'train_split1.csv'), 'w') as f:
        for i in range(len(train_json)):
            label = train_json[i]['template'].replace('[', '').replace(']', '').replace(' ', '_').replace(',', '_')
            f.write("%s.webm,%s,%s\n" % (train_json[i]['id'], cls_dict[label], label))
    print("=> train_split1.csv save at %s" % (os.path.join(ROOT_PATH, dataset_name, metafile_org)))

def video2frame(video_org_path,video_seq_path,file_info):
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
    meta_file_list = [[x.strip().split(',')[0],x.strip().split(',')[2]] for x in open(os.path.join(meta_file_path),'r')]

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
    create_dataset_dir()
    # somethingv2_process_metafile()
    # ## video to frame sequence
    # video2frame_pool(video2frame_video_org_path, video2frame_video_seq_path, video2frame_train_metafile)
    # video2frame_pool(video2frame_video_org_path, video2frame_video_seq_path, video2frame_val_metafile)
    ## generate frame sequence corresponding label
    video_seq_path = os.path.join(ROOT_PATH,dataset_name,video_seq)
    metafile_seq_path = os.path.join(ROOT_PATH,dataset_name,metafile_seq)
    for x in [(video2frame_train_metafile,'train_split1.csv'),(video2frame_val_metafile,'val_split1.csv')]:
        somethingv2_process_seq_metafile(x[0],
                                    video_seq_path,
                                    metafile_seq_path,
                                    save_name=x[1])
    print("Congratulations!!!")



