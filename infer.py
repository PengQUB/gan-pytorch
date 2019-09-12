# -*- coding: utf-8 -*-
import os
import time
import json
import torch
import numpy as np
import click
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import Resnet9, Resnet6

@click.command()
@click.option('--img_path', default='./datasets/0.png',
              prompt='Your image path:', help='image path')

def main(img_path):
    class Config(object):
        def __init__(self, j):
            self.__dict__ = json.load(j)

    use_gpu = False

    ckpt_path = './mic_gan-lsgan/'

    config_path = os.path.join(ckpt_path, 'config.json')
    ckpt_path = os.path.join(ckpt_path, 'netG.pt')

    config = Config(open(config_path, 'r'))

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,
                                                         std=std)
                                    ])
    size = config.val_image_size
    if isinstance(size, int):
        size = (int(size), int(size))
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img = img.resize(size, Image.BILINEAR)
    input = transform(img)

    model = Resnet9.ResnetGenerator(input_nc=config.in_channels,
                                    output_nc=config.out_channels,
                                    use_dropout=config.use_dropout,
                                    **config.model_params[config.G_model])
    model = torch.nn.DataParallel(model, use_gpu)
    if use_gpu:
        device = torch.device(use_gpu[0])
        input = input.to(device)
        model = model.to(device)
        ckpt = torch.load(ckpt_path)['net']
    else:
        ckpt = torch.load(ckpt_path, 'cpu')['net']

    model.load_state_dict(ckpt, strict=False)
    # state = torch.load(ckpt_path, 'cpu')
    model.eval()

    start = time.time()
    output = model(input.unsqueeze(0))
    print(f'Total forward time: {time.time()-start:.4f}s')

    predict_mask = transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))(output.squeeze(0))
    predict_mask = transforms.ToPILImage()(predict_mask)

    out = predict_mask.resize((w, h), Image.BILINEAR)
    out.save('./output.png')


    # fig, ax = plt.subplots(2)
    # ax[0].imshow(img.resize((w, h)))
    # ax[0].axis('off')
    # ax[1].imshow(predict_mask.resize((w, h)))
    # ax[1].axis('off')
    # plt.show()


if __name__ == '__main__':
    main()