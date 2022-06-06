# opt_ML_project


## Abstract

## Getting started

Requirements :
- PyTorch (>= 1.8)
- torchvision (>= 0.9)
- timm (>= 0.4.9)
- homura-core (>= 2021.3.1)

## Training Example on CIFAR10

We proposed a training example in which you can see the difference between ADAM optimizer, SAM and ASAM on CIFAR10.
You have the choice of using a small neural network (Net) or to use a state-of-the-art neural network on cifar10 : ResNet (wrn28_10)

**On Net :**

ADAM :
```
python run_cifar.py --dataset CIFAR10 --minimizer ADAM --epochs 200
```
SAM:
```
python run_cifar.py --dataset CIFAR10 --minimizer SAM --rho 0.05 --epochs 200
```
ASAM :
```
python run_cifar.py --dataset CIFAR10 --minimizer ASAM --rho 0.5 --epochs 200
```

**On wrn28_10 :**

ADAM :
```
python run_cifar.py --model wrn28_10 --dataset CIFAR10 --minimizer ADAM --epochs 100
```
SAM:
```
python run_cifar.py --model wrn28_10 --dataset CIFAR10 --minimizer SAM --rho 0.05 --epochs 100
```
ASAM :
```
python run_cifar.py --model wrn28_10 --dataset CIFAR10 --minimizer ASAM --rho 0.5 --epochs 100
```
