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

unet_params = {'filter_scale': 1}

unet_ae_params = {'backend': 'resnet18',
                  'pretrained': 'imagenet'}

deeplabv3_params = {'backend': 'resnet18',
                    'os': 16,
                    'pretrained': 'imagenet'}

rbpn_params = {'num_channels': 3,
               'base_filter': 256,
               'feat': 64,
               'num_stages': 3,
               'n_resblock': 5,
               }

ce_params = {'weight': None, 'ignore_index': 255}
dice_params = {'smooth': 1}
focal_params = {'weight': None, 'gamma': 2, 'alpha': 0.5}
lovasz_params = {'multiclasses': True}

parse = argparse.ArgumentParser(description='ImageSegmentation')

parse.add_argument('--model_params', default={'RBPN': rbpn_params,
                                              })

parse.add_argument('--loss_params', default={'ce': ce_params,
                                             'dice': dice_params,
                                             'focal': focal_params,
                                             'lovasz': lovasz_params})

parse.add_argument('--model', default='RBPN', choices=['RBPN', 'unet', 'unet_ae', 'dlv3plus', 'pspnet'], type=str)
parse.add_argument('--loss', default='ce', choices=['ce', 'dice', 'focal', 'lovasz'], type=str)
parse.add_argument('--lr', default=1e-4, type=float)
# parse.add_argument('--lr_decay_step', default=[30, 40], type=list)
# parse.add_argument('--lr_decay_rate', default=0.1, type=float)
parse.add_argument('--max_iters', default=None)
parse.add_argument('--epochs', default=150, type=int)
parse.add_argument('--batch_size', default=16 * 2, type=int)
parse.add_argument('--distributed', default=True, type=bool)
parse.add_argument('--gpuid', default='0,1,2,3,4,5,6,7', type=str)
parse.add_argument('--num_workers', default=16, type=int)
parse.add_argument('--ckpt_dir', default='./checkpoints/')
parse.add_argument('--resume', default=False, help='resume from checkpoint', type=bool)
parse.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parse.add_argument('--train_image_size', default=64, type=int, help='0 to use original frame size')
parse.add_argument('--val_image_size', default=(64, 64), help='w,h')
parse.add_argument('--in_channels', default=3, type=int)
parse.add_argument('--nFrames', default=7, type=int)
parse.add_argument('--upscale_factor', default=2, type=int)
parse.add_argument('--patch_size', default=64, type=int, help='0 to use original frame size')
parse.add_argument('--other_dataset', default=False, type=bool, help="use other dataset than vimeo-90k")
parse.add_argument('--future_frame', default=True, type=bool, help="use future frame")
parse.add_argument('--data_augmentation', default=False, type=bool)
parse.add_argument('--residual', default=False, type=bool)

## Vimeo90k path config
parse.add_argument('--data_type', default='vimeo90k', choices=['vimeo90k'])
parse.add_argument('--image_root', default='/gpu/zhangtong/resolution/tmp/vimeo_setuplet/sequences')
parse.add_argument('--train_list', default='../sep_trainlist.txt')
parse.add_argument('--val_list', default='../sep_testlist.txt')


def get_config():
    config, unparsed = parse.parse_known_args()
    config.ckpt_dir = os.path.join(config.ckpt_dir, f"{config.data_type}-{config.model}-{config.loss}")
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
