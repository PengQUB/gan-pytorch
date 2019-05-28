# -*- coding: utf-8 -*-
from .cyclegan import Resnet9, Resnet6
from .base_networks import Generator, Discriminator

ModelSelector = {'resnet_9blocks': Resnet9,
                 'resnet_6blocks': Resnet6
                 }
