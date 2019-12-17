import urllib.request
import rarfile           ## sudo apt install unrar | pip3 install rarfile
import sys
from multiprocessing import Pool
import cv2
import os
from functools import partial
'''
date:2019-12-09 14:52:12
author: jinyuanfeng
description: None
'''

DOWNLOAD_VIDEO = False
ROOT_PATH = "/data"       ##TODO: You only modify this flag according to your ROOT_PATH
dataset_name = 'HMDB51'
metafile_org = 'MetafileOrg'
metafile_seq = 'MetafileSeq'
video_org = 'VideoOrg'
video_seq = 'VideoSeq'
metafile_org_url = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'
video_orq_url = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'

video_org_save_path = os.path.join(ROOT_PATH,dataset_name,video_org,'hmdb51_org.rar')
metafile_org_save_path = os.path.join(ROOT_PATH,dataset_name,metafile_org,'test_train_splits.rar')

video2frame_train_metafile = os.path.join(ROOT_PATH,dataset_name,metafile_org,'train_split1.csv')
video2frame_val_metafile = os.path.join(ROOT_PATH,dataset_name,metafile_org,'val_split1.csv')
video2frame_video_org_path = os.path.join(ROOT_PATH,dataset_name,video_org,'hmdb51_org')
video2frame_video_seq_path = os.path.join(ROOT_PATH,dataset_name,video_seq)

def _progress(block_num, block_size, total_size):
    ''' callback func
       @block_num:  completed data block
       @block_size: size of data block
       @total_size: size of file
    '''
    sys.stdout.write('\r=> Downloading [%.1f%%]' % (float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

def create_dataset_dir():
    for x in [metafile_org,metafile_seq,video_org,video_seq]:
        path = os.path.join(ROOT_PATH, dataset_name, x)
        if os.path.isdir(path):
            print("=> %s has existed"%(path))
        else:
            print("=> %s will be created!"%(path))
            os.makedirs(path)
    print("=> create dir done!!!")

def download_dataset_metafile(download_url,save_path):
    print("=> %s will be saved \n"%(save_path))
    print("=> %s will download, downloading......"%(save_path))
    filepath, _ = urllib.request.urlretrieve(download_url, save_path, _progress)
    print("=> download done!!!")

def execute_rarfile(file_path,out_path=None):
    if not os.path.exists(file_path):
        print("=> file %s don't exist,please checking it"%(file_path))
        sys.exit(0)
    else:
        print("=> start uncompress rar file is %s"%(file_path))
        if out_path is None:
            out_path = os.path.dirname(file_path)
        rf = rarfile.RarFile(file_path)
        rf.extractall(out_path)
        print("=> uncompress done!")

def get_dirname(_path):
    return [x for x in os.listdir(_path) if os.path.isdir(os.path.join(_path,x))]

def  generate_file(PATH,split_type='split1',split_list=None):
    if not os.path.exists(PATH):
        print("=> %s don't exist,please checking it"%(PATH))
        sys.exit(0)
    train_list = []
    val_list = []
    test_list = []
    cls_dict = {}
    cnt = -1
    for x in split_list:
        x_item = x.strip().split('_test_'+split_type+'.txt')
        cnt += 1
        cls_dict[x_item[0]] = cnt
        _path = os.path.join(PATH,x)
        if not os.path.exists(_path):
            print("=> %s don't exist,please checking it" % (_path))
            sys.exit(0)
        for f in open(_path, 'r'):
            item = f.strip().split(' ')
            if int(item[1]) == 1:  # train
                train_list.append((x_item[0]+'/'+item[0], cnt, x_item[0]))
            if int(item[1]) == 2:  # val
                val_list.append((x_item[0]+'/'+item[0], cnt, x_item[0]))
            if int(item[1]) == 0:  # test
                test_list.append((x_item[0]+'/'+item[0], cnt, x_item[0]))
    print("=> hmdb51 %s train len is %d"%(split_type,len(train_list)))
    print("=> hmdb51 %s val len is %d"%(split_type,len(val_list)))
    print("=> hmdb51 %s test len is %d"%(split_type,len(test_list)))
    print("=> hmdb51 class dict is",cls_dict)

    save_path = os.path.join(ROOT_PATH,dataset_name,metafile_org)
    file_name = os.path.join(save_path,"train_"+split_type+".csv")
    with open(file_name, 'w') as f:
        for x in train_list:
            f.write("%s,%s,%s\n"%(x[0],str(x[1]),x[2]))
    file_name = os.path.join(save_path,"val_"+split_type+".csv")
    with open(file_name, 'w') as f:
        for x in val_list:
            f.write("%s,%s,%s\n" % (x[0], str(x[1]), x[2]))
    file_name = os.path.join(save_path,"test_"+split_type+".csv")
    with open(file_name, 'w') as f:
        for x in test_list:
            f.write("%s,%s,%s\n" % (x[0], str(x[1]), x[2]))
    file_name = os.path.join(save_path,"cls_"+split_type+".csv")
    with open(file_name, 'w') as f:
        for x in cls_dict.keys():
            f.write("%s,%s\n"%(str(cls_dict[x]),x))

def hmdb51_process_metafile(_path):
    dirname = get_dirname(_path)
    if len(dirname) == 1:
        file_list = os.listdir(os.path.join(ROOT_PATH, dataset_name,metafile_org,dirname[0]))
        print(len(file_list))
        split1 = []
        split2 = []
        split3 = []
        for x in file_list:
            if 'split1' in x:
                split1.append(x)
            if 'split2' in x:
                split2.append(x)
            if 'split3' in x:
                split3.append(x)
        _path = os.path.join(ROOT_PATH, dataset_name, metafile_org, dirname[0])
        generate_file(_path, split_type='split1', split_list=split1)
        generate_file(_path, split_type='split2', split_list=split2)
        generate_file(_path, split_type='split3', split_list=split3)
        print("=> metafile_org done")
    else:
        print("=> dir format error")
        print("=> maybe you need to uncompress hmdb51 metafile_org.rar")
        sys.exit(0)

def video2frame(video_org_path,video_seq_path,file_name):
    '''
    video_org_path: format: video_org_dir/
                    structure: video_org_dir/class_dir/video_name.avi['mp4']
    video_seq_path: format: video_seq_dir/
                    structure: video_seq_dir/class_dir/video_name_dir/
    file_name:  format:  class_dir/video_name.avi['mp4']
    info.txt: format: frame number,frame rate, duration,height,width
    '''
    print("=> %s will be processed"%(file_name))
    file_path = os.path.join(video_org_path,file_name)
    save_frame_path = os.path.join(video_seq_path,file_name.split(".")[0])
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
    meta_file_list = [x.strip().split(',')[0] for x in open(os.path.join(meta_file_path),'r')]
    pool = Pool(10)
    p_video2frame = partial(video2frame,video_org_path,video_seq_path)
    pool.map_async(p_video2frame,meta_file_list)
    pool.close()
    pool.join()
    print("=> done!!!")

def hmdb51_process_seq_metafile(metafile_org_path,
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
        video_name = item[0].strip().split('.')[0]
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
    if not DOWNLOAD_VIDEO:
        print("=> you have selected download mode:False!!!")
        print("=> you should download original label")
        print("=> URL: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar")
        print("=> you should put it into /your_root_path/HMDB51/MetafileOrg/,and uncompress it")
        print("=> you should download original video")
        print("=> URL: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar")
        print("=> you should put it into /your_root_path/HMDB51/VideoOrg/, and uncompress it")
        print("=> when you completed the above requirements, you can try run it again!!!")
        if not os.path.exists(metafile_org_save_path):
            print("=> %s don't exist,please completing the above work"%(metafile_org_save_path))
            sys.exit(0)
        if not os.path.exists(video_org_save_path):
            print("=> %s don't exist,please completing the above work" % (video_org_save_path))
            sys.exit(0)

    if DOWNLOAD_VIDEO:
        download_dataset_metafile(metafile_org_url,metafile_org_save_path)
        execute_rarfile(metafile_org_save_path)
    path = os.path.join(ROOT_PATH,dataset_name,metafile_org)
    hmdb51_process_metafile(path)
    if DOWNLOAD_VIDEO:
        download_dataset_metafile(video_orq_url,video_org_save_path)
        execute_rarfile(video_org_save_path)
    ## video to frame sequence
    video2frame_pool(video2frame_video_org_path, video2frame_video_seq_path, video2frame_train_metafile)
    video2frame_pool(video2frame_video_org_path, video2frame_video_seq_path, video2frame_val_metafile)
    ## generate frame sequence corresponding label
    video_seq_path = os.path.join(ROOT_PATH,dataset_name,video_seq)
    metafile_seq_path = os.path.join(ROOT_PATH,dataset_name,metafile_seq)
    for x in [(video2frame_train_metafile,'train_split1.csv'),(video2frame_val_metafile,'val_split1.csv')]:
        hmdb51_process_seq_metafile(x[0],
                                    video_seq_path,
                                    metafile_seq_path,
                                    save_name=x[1])
    print("Congratulations!!!")



