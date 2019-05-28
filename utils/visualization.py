# -*- coding: utf-8 -*-
import torch
import numpy as np

def vis_seq(sr_outputs):
    rgb_imgs = []
    for sr_img in sr_outputs:
        rgb_imgs.append(sr_img)
    return torch.from_numpy(np.array(rgb_imgs))