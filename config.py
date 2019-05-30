# -*- coding: utf-8 -*-
import os
import logging
import argparse

'''
All backbones
['resnet18', 'resnet34', 'resnet50', 'resnet101',
 'resnet152', 'senet154', 'se_resnet50', 'se_resnet101',
 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
 'mobilenet_v2', 'shufflenet_v2'(for deeplabv3+)]
'''

resnet9_params = {'n_blocks': 9}

resnet6_params = {'n_blocks': 6}

ce_params = {'weight': None, 'ignore_index': 255}
dice_params = {'smooth': 1}
focal_params = {'weight': None, 'gamma': 2, 'alpha': 0.5}
lovasz_params = {'multiclasses': True}

parse = argparse.ArgumentParser(description='cyclegan')

parse.add_argument('--model_params', default={'resnet_9blocks': resnet9_params,
                                              'resnet_6blocks': resnet6_params
                                              })

parse.add_argument('--loss_params', default={'ce': ce_params,
                                             'dice': dice_params,
                                             'focal': focal_params,
                                             'lovasz': lovasz_params})

parse.add_argument('--image_root', default='/gpu/zhangtong/gender/FtoM')

parse.add_argument('--gan_mode', default='lsgan', choices=['vanilla', 'lsgan', 'wgangp'], type=str)
parse.add_argument('--model_name', default='f2m_cyclegan', type=str)
parse.add_argument('--G_model', default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet128', 'unet256'], type=str)
parse.add_argument('--D_model', default='n_layer', choices=['n_layers', 'pixel'], type=str)
parse.add_argument('--loss', default='ce', choices=['ce', 'dice', 'focal', 'lovasz'], type=str)
parse.add_argument('--lr', default=5e-3, type=float)
# parse.add_argument('--lr_decay_step', default=[30, 40], type=list)
# parse.add_argument('--lr_decay_rate', default=0.1, type=float)
parse.add_argument('--max_iters', default=None)
parse.add_argument('--epochs', default=300, type=int)
parse.add_argument('--batch_size', default=1, type=int)
parse.add_argument('--distributed', default=True, type=bool)
parse.add_argument('--gpuid', default='0,1,2,3,4,5,6,7', type=str)
parse.add_argument('--num_workers', default=16, type=int)
parse.add_argument('--ckpt_dir', default='./checkpoints/')
parse.add_argument('--resume', default=False, help='resume from checkpoint', type=bool)
parse.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parse.add_argument('--train_image_size', default=512, type=int, help='0 to use original frame size')
parse.add_argument('--val_image_size', default=(512, 512), help='w,h')
parse.add_argument('--in_channels', default=3, type=int)
parse.add_argument('--out_channels', default=3, type=int)
parse.add_argument('--use_dropout', default=False, type=bool)
parse.add_argument('--unaligned', default=False, type=bool)



def get_config():
    config, unparsed = parse.parse_known_args()
    config.ckpt_dir = os.path.join(config.ckpt_dir, f"{config.model_name}-{config.gan_mode}")
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    logger = logging.getLogger("InfoLog")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(config.ckpt_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return config
