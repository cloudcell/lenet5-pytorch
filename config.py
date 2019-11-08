from yacs.config import CfgNode as CN

_C = CN()

_C.DEVICE = 'cuda'

_C.PATHS = CN()
_C.PATHS.DATASET = '/Users/maorshutman/data/FashionMNIST'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 1.0e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.EPOCHS = 20
_C.TRAIN.STEPS_PER_EPOCH = 200
_C.TRAIN.BATCHNORM = False
_C.TRAIN.DROPOUT = False
_C.TRAIN.LOG_INTERVAL = 30
_C.TRAIN.LOG_DIR = './logs'

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32
_C.TEST.STEPS = 200


cfg = _C


def load_from_yaml(path):
    import yaml
    with open(path, 'r') as f:
        cfg_dict = yaml.load(f)
    tmp = CN(cfg_dict)
    cfg.merge_from_other_cfg(tmp)
