import imp
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

def batchnorm_layer(nb_features, nb_dim):
    assert nb_dim ==1 or nb_dim ==2 or nb_dim==3
    if nb_dim == 2:
        return nn.BatchNorm2d(nb_features)
    elif nb_dim == 3:
        return nn.BatchNorm3d(nb_features)
    else:
        return nn.BatchNorm1d(nb_features)

def conv_layer(nb_dim, use_spectral_norm, *args, **kwargs):
    assert nb_dim == 1 or nb_dim == 2 or nb_dim == 3
    
    if nb_dim == 2:
        l = nn.Conv2d(*args, **kwargs)
    elif nb_dim == 3:
        l = nn.Conv3d(*args, **kwargs)
    else:
        l = nn.Conv1d(*args, **kwargs)
    if use_spectral_norm:
        return spectral_norm(l)
    else:
        return l

def maxpool(kernel_size, nb_dim):
    assert nb_dim == 2 or nb_dim == 3
    if nb_dim == 2:
        return nn.MaxPool2d(kernel_size=kernel_size)
    else:
        return nn.MaxPool3d(kernel_size=kernel_size)

def avgpool(kernel_size, nb_dim):
    assert nb_dim == 2 or nb_dim == 3
    if nb_dim == 2:
        return nn.AvgPool2d(kernel_size=kernel_size)
    else:
        return nn.AvgPool3d(kernel_size=kernel_size)

def linear(feature_in, features_out, use_spectral_norm):
    if not use_spectral_norm:
        return nn.Linear(feature_in, features_out)
    else:
        return spectral_norm(nn.Linear(feature_in, features_out))
