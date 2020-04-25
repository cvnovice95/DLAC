import os
import time
from easydict import EasyDict as edict

__C = edict()
ActivityConfig = __C

__C.TIMESTAMP = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime()) # To initialize timestamp when importing this file
__C.WORKSPACE_PATH = os.getcwd()   # To get current workspace path
__C.GPUS = 8                       # Setting the number of GPU
__C.USE_SUMMARY = False             # To save log of tensorboard. If you need it,the flag should be set True.
__C.SUPER_LOADER = False           # TODO: If you have sufficient CPU and Memory, the flag is used to accelerate dataloader.
__C.USE_DISTRIBUTED = False        # TODO: If you set it, the training will use DistributedDataParallel.
__C.USE_SYNC_BN = True if __C.USE_DISTRIBUTED else False # If and only if USE_DISTRIBUTED is set True, it be marked True
__C.USE_APEX = False               # Using apex lib to accelerate training, URL:https://github.com/NVIDIA/apex
__C.KEEP_HISTORY = True            # 1 keep 0 remove | if this flag be marked True, framework will preserve history log
__C.DEBUG_MODE = False             # 1 debug mode 0 normal | if this flag be marked True, framework will save log into debug[dir]
## Experiment Setting
__C.EXP_TYPE = 'T'                 # TODO: the type of experiment,['T','E')
__C.MODEL_NAME = 'tsn'             # TODO: the model name of experiment,['tsn','i3d','slowfast','tsm']
__C.MODEL_LIST = ['tsn','i3d','slowfast','tsm','mfnet3d']
__C.EXP_TAG = 'sv2_BNInception_RGB'     # TODO: the flag of experiment
__C.BACKBONE = 'BNInception'       # TODO: backbone network ['BNInception','mobilenetv2','resnet50','mfnet2d','inceptionv3']
__C.PRETRAIN_TYPE = 'imagenet'     # backbone pretrain type ['imagenet','kinetics',None]
__C.PRETRAIN_TYPE_LIST = ['imagenet','kinetics']
__C.RESUME_TYPE = 'resume'         # ['resume','finetune'] 'resume' is used to resume checkpoint,
                                                        #  'finetune' is used to load other model's params
## Experiment Folder Setting
__C.SNAPSHOT_ROOT = "/data"        # To set root path which saves output of framework
__C.EXP_NAME = "{}_{}_{}".format(__C.EXP_TYPE,__C.MODEL_NAME,__C.EXP_TAG)
__C.SNAPSHOT_LOG = os.path.join(__C.SNAPSHOT_ROOT,'ar_output',__C.EXP_NAME,'log')
__C.SNAPSHOT_LOG_DEBUG = os.path.join(__C.SNAPSHOT_ROOT,'ar_output',__C.EXP_NAME,'log','debug')
__C.SNAPSHOT_CHECKPOINT = os.path.join(__C.SNAPSHOT_ROOT,'ar_output',__C.EXP_NAME,'checkpoint')
__C.SNAPSHOT_SUMMARY = os.path.join(__C.SNAPSHOT_ROOT,'ar_output',__C.EXP_NAME,'summary')
__C.SNAPSHOT_CONFIG = os.path.join(__C.SNAPSHOT_ROOT,'ar_output',__C.EXP_NAME,'config')
__C.PRETRAIN_MODEL_ZOO = os.path.join(__C.SNAPSHOT_ROOT,'ar_output','pretrain_model_zoo')  # save backbone pretrain model params
## Dataset Configuration
# root_path/
#          |-HDMDB51/
#                   |-VideoSeq/{class_dir}/{frame_sequence}
#                   |-VideoOrg/
#                   |-MetafileSeq/
#                   |-MetafileOrg/
__C.DATASET = edict()
__C.DATASET.ROOT_PATH = "/data" # The root path of Dataset
__C.DATASET.NAME = "SomethingV2"     # The name of Dataset
__C.DATASET.CLASS_NUM = 174      # TODO: CLASS_NUM
__C.DATASET.IMG_FORMART = 'img_{:05d}.jpg'
__C.DATASET.VIDEO_SEQ_PATH = os.path.join(__C.DATASET.ROOT_PATH,__C.DATASET.NAME,'VideoSeq')
__C.DATASET.TRAIN_META_PATH = os.path.join(__C.DATASET.ROOT_PATH,__C.DATASET.NAME,'MetafileSeq','train_split1.csv')
__C.DATASET.VAL_META_PATH = os.path.join(__C.DATASET.ROOT_PATH,__C.DATASET.NAME,'MetafileSeq','val_split1.csv')
__C.DATASET.TRAIN_SAMPLE_METHOD = 'seg_random'
__C.DATASET.VAL_SAMPLE_METHOD = 'seg_ratio'
__C.DATASET.LOAD_METHOD = 'TSNDataSet'
__C.DATASET.LOAD_METHOD_LIST = ['TSNDataSet',
                                'DPFlowDataset',
                                'HumanactionDataset',
                                'DPFlowUntrimmedDataset']
## Training Configuration
__C.TRAIN = edict()
__C.TRAIN.EPOCHS = 80            # The number of Epoch
__C.TRAIN.BASE_BATCH_SIZE = 64    # TODO: BASE_BATCH_SIZE
__C.TRAIN.BATCH_SIZE = __C.TRAIN.BASE_BATCH_SIZE//__C.GPUS if __C.USE_DISTRIBUTED else __C.TRAIN.BASE_BATCH_SIZE   # mini batch size
__C.TRAIN.BASE_LR = 0.001        # The base learning rate of the whole training
__C.TRAIN.LR_STEP_TYPE = 'step'  # The policy of learning rate, ['step','sgdr']
__C.TRAIN.LR_STEPS =[20,40,60]      # If LR_SETP_TYPE = 'step', the change step of the learning rate
__C.TRAIN.PERIOD_EPOCH = 10      # To set period params in SGDR
__C.TRAIN.WARMUP_EPOCH = 10      # To set warmup params in SGDR
__C.TRAIN.MOMENTUM = 0.9         # To set momentum params
__C.TRAIN.WEIGHT_DECAY = 5e-4    # To set weight decay params
__C.TRAIN.CLIP_GRADIENT = 20     # To set clip gradient threshold
__C.TRAIN.PRINT_FREQ = 4         # To set log.info print frequency
__C.TRAIN.EVALUATE_FREQ = 5      # To set save checkpoint frequency
__C.TRAIN.START_EPOCH = 0        # To set start epoch
__C.TRAIN.BEST_PREC = 0.0        # To set global best top1 in training
__C.TRAIN.DROPOUT = 0.8          # To set dropout
__C.TRAIN.SEG_NUM = 16            # TODO: set seg num in TSN sampler policy
__C.TRAIN.CROP_NUM = 1
__C.TRAIN.MODALITY = 'RGB'       # To set data modality
__C.TRAIN.PARTIAL_BN = True      # To set partial bn in TSN
### TSM params
__C.TRAIN.IS_SHIFT = True        # To set shift in TSM
__C.TRAIN.SHIFT_DIV = 8          # To set shift ratio of channel in TSM
__C.TRAIN.SHIFT_PLACE = 'block' if __C.MODEL_NAME == 'tsm' and __C.BACKBONE == 'BNInception' else 'blockres'
## Validation Configuration
__C.VAL = edict()
__C.VAL.BATCH_SIZE = 1           # To set batch size in evaluate
__C.VAL.PRINT_FREQ = 1           # To set log.info print frequency
__C.VAL.SEG_NUM = 16             # TODO: set seg num in evaluate
__C.VAL.CROP_NUM = 1
__C.VAL.MODALITY = 'RGB'         # To set data modality in evaluate
## The info of input's image
__C.IMG = edict()
__C.IMG.CROP_SIZE = 224          # To set input img size
__C.IMG.SCALE_SIZE = __C.IMG.CROP_SIZE * 256 // 224
__C.IMG.MEAN = 0
__C.IMG.STD = 0
## PRETRAIN_MODEL_DICT URL:
__C.PRETRAIN_MODEL_DICT = {
    'tsn':{
        'BNInception':{
            'imagenet':'bn_inception-52deb4733.pth',
            'resume':'',
            'finetune':''
        },
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        },
        'inceptionv3':{
            'imagenet':'inception_v3_google-1a9a5a14.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'i3d':{
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'slowfast':{
        'resnet50':{
            'imagenet':'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        }
    },
    'tsm':{
        'resnet50':{
            'imagenet': 'resnet50-19c8e357.pth',
            'resume':'',
            'finetune': ''
        },
        'BNInception':{
            'imagenet': 'BNInception-9baff57459f5a1744.pth',
            'kinetics': 'BNInceptionKinetics-47f0695e.pth',
            'resume':'',
            'finetune': ''
        },
        'mobilenetv2':{
            'imagenet':'mobilenetv2_1.0-f2a8633.pth.tar',
            'resume':'',
            'finetune': ''
        }
    },
    'mfnet3d':{
        'mfnet2d':{
            'imagenet': 'MFNet2D_ImageNet1k-0000.pth',
            'resume':'',
            'finetune': ''
        }
    }
}
__C.TRAIN.RESUME = __C.PRETRAIN_MODEL_DICT[__C.MODEL_NAME][__C.BACKBONE][__C.RESUME_TYPE]    #_C.SNAPSHOT_CHECKPOINT + pth name
__C.TRAIN.PRETRAIN_MODEL = __C.PRETRAIN_MODEL_DICT[__C.MODEL_NAME][__C.BACKBONE][__C.PRETRAIN_TYPE]# _C.PRETRAIN_MODEL_ZOO + C.TRAIN.PRETRAIN_MODEL

## DPFlow and Redis Configuration
__C.REDIS_MODE = False   # TODO:
__C.REDIS_PATH = '/data/datasets/redis/'
__C.REDIS_DATASET = 'kinetics600'          #'activitynet_resized_trimmed'
__C.REDIS_LABEL_PKL = 'kinetics600_label_to_label_id.pkl'        #'activitynet_trimmed_label_to_label_id.pkl'
__C.REDIS_DATASET_CLASS_NUM = __C.DATASET.CLASS_NUM
__C.REDIS_TRAIN_SAMPLE_METHOD = 'seg_random'
__C.REDIS_VAL_SAMPLE_METHOD = 'seg_ratio'

__C.REDIS_TRAIN_SEG_NUM = __C.TRAIN.SEG_NUM
__C.REDIS_TRAIN_CROP_NUM = __C.TRAIN.CROP_NUM
__C.REDIS_TRAIN_BATCH_SIZE = __C.TRAIN.BATCH_SIZE

__C.REDIS_VAL_SEG_NUM = __C.VAL.SEG_NUM
__C.REDIS_VAL_CROP_NUM = __C.VAL.CROP_NUM
__C.REDIS_VAL_BATCH_SIZE = __C.VAL.BATCH_SIZE
## Humanaction Dataset
__C.HUMANACTION_PKL_ROOT_PATH = '/data/humanaction'
__C.HUMANACTION_TRAIN_PKL_NAME = 'humanaction_training.pkl'
__C.HUMANACTION_VAL_PKL_NAME = 'action_benchmark_s1_attversion.pkl'
__C.HUMANACTION_LABEL_LIST = [
    'static_action_fall',
    'static_action_run',
    'static_action_smoke',
    'static_action_use_phone',
    'static_action_watch_phone',
    # 'unclear'
]




