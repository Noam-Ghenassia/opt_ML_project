# opt_ML_project


## Abstract

## Getting started

Requirements :
- PyTorch (>= 1.8)
- torchvision (>= 0.9)
- timm (>= 0.4.9)
- homura-core (>= 2021.3.1)

## Training Example on CIFAR10

To explore a concrete example of ASAM efficiency, we reproduced the training of ResNet (wrn28_10) on cifar10 using ASAM proposed by SAMSUNG.
To compare its results with other optimizers, we proposed 3 possible choices : ADAM, ASAM or SAM.
As ResNet may be hard to train, we also created a small neural network (called Net) able to train on cifar10, but using only few layers.

**On Net :**

ADAM :
```
python run_cifar.py --minimizer ADAM --epochs 200
```
SAM:
```
python run_cifar.py --minimizer SAM --rho 0.05 --epochs 200
```
ASAM :
```
python run_cifar.py --minimizer ASAM --rho 0.5 --epochs 200
```

**On wrn28_10 :**

ADAM :
```
python run_cifar.py --model wrn28_10 --minimizer ADAM --epochs 100
```
SAM:
```
python run_cifar.py --model wrn28_10 --minimizer SAM --rho 0.05 --epochs 100
```
ASAM :
```
python run_cifar.py --model wrn28_10 --minimizer ASAM --rho 0.5 --epochs 100
```
