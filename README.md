# lenet5-pytorch
This repository contains an implemenation of the Lenet5 architecture. The original subsampling and C3 convolution layers where implemented according to the orginal paper: *LeCun, Yann, et al. "Object recognition with gradient-based learning." 1999*. The LeNet5 model implemented here allows to use more modern layes and activations like MaxPooling, and ReLU.
 
## Train
```bash
python train.py \
  --cfg_path=./cfg_files/cfg.yaml \
  --hp_optim=False 
```

## Test
```bash
python test.py \
  --checkpoint_path=./checkpoints/checkpoint_19.pt \
  --model_path=./checkpoints/model.pt \
  --cfg_path=./cfg_files/cfg.yaml
```

## Performance
We train the model on the FashionMNIST dataset. The model reaches an accuracy of 88.4% on the test set, and 90.4% on the validaiton set.
Bellow we have a confusion matrix on the validation set for last epoch. The configuration used is ./cfg_file/cfg.yaml.

![conf mat](/logs/run_1/conf_mat_14080.png)
