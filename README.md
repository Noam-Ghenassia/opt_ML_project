# Adaptive Sharpness-Aware Minimization (ASAM) for Scale-Invariant Learning of Deep Neural Networks
Damien Gomez Donoso, Noam Ghenassia, Théau Vannier

## Abstract
To train robust models, we need a way of avoiding the optimization process to end in a minimum that does not generalize. This is usually the case of sharper minima. Indeed, one should keep in mind that due to finite size effects, the training loss landscape is not exactly equal to that of the actual distribution. As a result, sharper minima of the training loss function yield higher test loss values. 
SAM idea is to constrained a sharpness term while minimizing the loss. However, minimizing the loss of a neural network can yield many combinations of weights that that give rise to the same neural network function. Some of these combinations of weights can have different sharpness terms. It has been shown that in this case, the SAM loss is falsely lead by the sharpness term.
ASAM is an extension of the SAM minimizer that tries to take into account this equivalence of parameter vectors in its minimization process. To do so, it applies a transformation on the sharpness term such that it stays constant for parameter vectors that support the same function

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
## Training on toy datsets

Test the generalization of ASAM and SAM on simple datasets and compare them with the generalzation of ADAM optimizer. For this test, we construct a toy dataset drawn from a known distribution whose variance can be controlled. We train three different neural networks on the same toy dataset with a certain variance (--var) and with a fixed number of data points (--number_points). Once trained, we test these neural networks on dataset with different variances and we compare the test errors. These errors are reported on a graph (the output of the function : test_performance_ADAM_ASAM_SAM).

```
python run_toy_datasets.py --number_points 2000 --var 10 --epochs 200
```
