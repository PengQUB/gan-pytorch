# -*- coding: utf-8 -*-
from .generator import Resnet9, Resnet6
from .discriminator import Nlayer, Pixel
from .base_networks import Generator, Discriminator

GModelSelector = {'resnet_9blocks': Resnet9,
                 'resnet_6blocks': Resnet6
                 }

DModelSelector = {'n_layer': Nlayer,
                 'pixel': Pixel
                 }
