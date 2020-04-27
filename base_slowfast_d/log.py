import os
import logging
from config import ActivityConfig as cfg
from utils import IO


def _loger():
    if cfg.DEBUG_MODE:
        if not cfg.KEEP_HISTORY:
            _path = os.path.join(cfg.SNAPSHOT_LOG_DEBUG)
            IO.del_file(_path)
        name = os.path.join(cfg.SNAPSHOT_LOG_DEBUG,cfg.TIMESTAMP + "_DEBUG.log")
    else:
        if not cfg.KEEP_HISTORY:
            _path = os.path.join(cfg.SNAPSHOT_LOG)
            IO.del_file(_path)
        name =  os.path.join(cfg.SNAPSHOT_LOG,cfg.TIMESTAMP +".log")
    logger = logging.getLogger("log")
    # logger.disabled = True
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=name,mode='w')
    logger.setLevel(logging.DEBUG)
    # handler1.setLevel(logging.DEBUG)
    # handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(filename)s[line:%(lineno)d]-%(module)s-%(funcName)s-%(levelname)s: %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
loger = _loger()