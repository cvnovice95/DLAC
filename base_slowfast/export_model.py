from config import ActivityConfig as cfg
import sys
import os
import pprint as pp
import pickle as pkl
import shutil
def export_model(output_path=None):
    export_path = "./"
    if 'T' not in cfg.EXP_TYPE:
        print("=> Your Exp Type is %s, not 'T' " % (cfg.EXP_TYPE))
        sys.exit(0)
    print("=> %s 's params of model and config of model will be exproted!!!"%(cfg.EXP_NAME))
    if output_path is None:
        print("=> You don't set export path,will use default path %s"%(cfg.PRETRAIN_MODEL_ZOO))
        if os.path.exists(cfg.PRETRAIN_MODEL_ZOO):
            print("=> %s is existed" % (cfg.PRETRAIN_MODEL_ZOO))
        else:
            print("=> %s is not existed" % (cfg.PRETRAIN_MODEL_ZOO))
            print("=> %s will be created" % (cfg.PRETRAIN_MODEL_ZOO))
            os.makedirs(cfg.PRETRAIN_MODEL_ZOO)
        export_path = cfg.PRETRAIN_MODEL_ZOO
    else:
        print("=> export path is %s"%(output_path))
        if not os.path.exists(output_path):
            print("=> %s don't exist!!!,Please checking it"%(output_path))
            sys.exit(0)
        export_path = output_path
    exp_name = cfg.EXP_NAME
    checkpoint_path = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', exp_name, 'checkpoint')
    config_path = os.path.join(cfg.SNAPSHOT_ROOT, 'ar_output', exp_name, 'config')
    checkpoint_name = None

    if not os.path.exists(config_path):
        print("=> %s is not exists!!!!!!" % (config_path))
        sys.exit(0)
    if not os.path.exists(checkpoint_path):
        print("=> %s is not exists!!!!!!" % (checkpoint_path))
        sys.exit(0)

    lst = os.listdir(checkpoint_path)
    lst = sorted(lst)
    if len(lst) == 0:
        print("=> don't have any checkpoint file!!!")
        print("=> Please finishing training once!!!")
        sys.exit(0)
    else:
        ans = input("=>only show best checkpoint file?[y/n]:")
        if ans in ['y', 'n', 'Y', 'N']:
            if ans in ['y', 'Y']:
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
                print("=> exporting checkpoint %s" % (checkpoint_file))
                config_name = checkpoint_name[:19] + "_cfg.pkl"
                config_path = os.path.join(config_path, config_name)
                if not os.path.exists(config_path):
                    print("=> %s don't exist!!!!,Please checking it." % (config_path))
                    sys.exit(0)
                print("=> exporting config %s" % (config_path))
                with open(config_path, 'rb') as f:
                    cfg_dict = pkl.load(f)
                cfg.__dict__.update(cfg_dict)
                export_name_pth = "{}_{}_Seg{}_{}_{}.pth".format(cfg.MODEL_NAME,cfg.BACKBONE,cfg.TRAIN.SEG_NUM,cfg.TRAIN.MODALITY,cfg.DATASET.NAME)
                export_name_pkl = "{}_{}_Seg{}_{}_{}.pkl".format(cfg.MODEL_NAME,cfg.BACKBONE,cfg.TRAIN.SEG_NUM,cfg.TRAIN.MODALITY,cfg.DATASET.NAME)
                export_path_pth = os.path.join(export_path,export_name_pth)
                export_path_pkl = os.path.join(export_path,export_name_pkl)
                shutil.copyfile(config_path, export_path_pkl)
                print("export config file to %s " % (export_path_pkl))
                shutil.copyfile(checkpoint_file, export_path_pth)
                print("export model file to %s "% (export_path_pth))

        else:
            print("your number is not in range!!!")
            sys.exit(0)

if __name__ == '__main__':
    export_model(output_path=None)
