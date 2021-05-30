# Point-based model
**Learning to Solve Parametric Partial Differential Equations through Point-based Neural Networks**

Ning Hua

## Introduction
This project is based on our paper -- Learning to Solve Parametric Partial Differential Equations through Point-based Neural Networks. 

## Installation
The code is based on [PointConv](https://github.com/DylanWusee/pointconv)

The code has been tested with Python 3.6.3, TensorFlow 1.12.0, cuda 9.1 and libcudnn 7 on Ubuntu 16.04.

## Usage
### Poisson's Equation

```
python train_poisson.py
```
```
python test_poisson.py
```

### Darcy Flow Equation
```
python train_darcyflow.py
```
```
python test_darcyflow.py
```

Modify the model_path to your .ckpt file path and the data_path to the .npy file path for Poisson's Equation and .mat for Darcy Flow Equation (provided in https://github.com/zongyi-li/fourier_neural_operator).

