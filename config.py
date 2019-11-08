from yacs.config import CfgNode as CN

_C = CN()

_C.DEVICE = 'cuda'

_C.PATHS = CN()
_C.PATHS.DATA = '/Users/maorshutman/data/FashionMNIST'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 1.0e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCHNORM = False
_C.TRAIN.DROPOUT = False
_C.TRAIN.LOG_INTERVAL = 30
_C.TRAIN.LOG_DIR = './logs'

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32


cfg = _C
