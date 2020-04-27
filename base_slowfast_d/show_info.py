import os
import sys
import pprint as pp
from config import ActivityConfig as cfg

def show_info():
    if not os.path.exists(cfg.SNAPSHOT_CONFIG):
        print("=> %s don't exist!!!,Please checking it"%(cfg.SNAPSHOT_CONFIG))
        sys.exit(0)
    lst = os.listdir(cfg.SNAPSHOT_CONFIG)
    if len(lst) == 0:
        print("=> don't have any config file!!!")
        sys.exit(0)
    else:
        opt_dict = {}
        for i in range(len(lst)):
            opt_dict[str(i)] = lst[i]
        opts = sorted(opt_dict.items(), key=lambda d: int(d[0]))
        pp.pprint(opts)
        # pp.pprint(opt_dict)
        num = input("=> please input number to show file info:")
        if str(num) in list(opt_dict.keys()):
            _config_path = os.path.join(cfg.SNAPSHOT_CONFIG, opt_dict[str(num)])
            with open(_config_path,"r") as f:
                print(f.read())
            prefix = opt_dict[str(num)].split("_cfg.txt")[0]
            # check log
            _log_path = os.path.join(cfg.SNAPSHOT_LOG,"{}.log".format(prefix))
            if os.path.exists(_log_path):
                with open(_log_path, "r") as f:
                    print(f.read())
            else:
                print("=> log %s is not exists"%(_log_path))
        else:
            print("your number is not in range!!!")
            sys.exit(0)
if __name__ == '__main__':
    show_info()