import os
import yaml
from easydict import EasyDict as edict
import numpy as np
import logging
import time
import sys
from utils.misc import copy_all_code


def Set_Config(args):
    # get config file path
    cfg_path = args.cfg_path

    # load config file
    f = open(cfg_path, 'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    cfg = edict(x)

    return cfg


def Set_Logger(args, cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # import pdb
    # pdb.set_trace()

    if args.local_rank == 0:
        if cfg.save_log:
            time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime(time.time() + 8 * 60 * 60))
            assert cfg.save_path != 'None'
            if not os.path.exists(cfg.save_path):
                os.makedirs(cfg.save_path)
            if args.mode == 'eval':
                if 'resume_all_dir' in cfg.keys() and cfg.resume_all_dir != 'None':
                    log_name = 'log_valALL_%s_%s.log' % (cfg.dataset.name, time_str)
                else:
                    log_name = 'log_val_%s_%s.log' % (cfg.dataset.name, time_str)
            elif args.mode == 'train':
                log_name = 'log_train_%s_%s.log' % (cfg.dataset.name, time_str)
            else:
                exit()

            fh = logging.FileHandler(filename=os.path.join(cfg.save_path, log_name), mode='a', encoding=None,
                                     delay=False)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler(stream=sys.stdout)
            logger.addHandler(ch)

    return logger


def Set_Ckpt_Code_Debug_Dir(cfg, args, logger, include_dir=['configs', 'utils', 'models']):
    checkpoint_dir = None
    if cfg.save_model and args.local_rank == 0:
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)

        checkpoint_dir = cfg.save_path + '/ckpts/'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            error_msg = 'checkpoint_dir already exist'
            logger.error(error_msg)
            raise ValueError(error_msg)

        code_dir = os.path.join(cfg.save_path, 'code')
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        else:
            error_msg = 'code_dir already exist: %s' % (code_dir)
            logger.error(error_msg)
            raise ValueError(error_msg)
        copy_all_code('./', code_dir, include_dir=include_dir)

        debug_dir = os.path.join(cfg.save_path, 'debug')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    elif args.local_rank == 0:
        debug_dir = os.path.join('./', 'debug')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    return checkpoint_dir