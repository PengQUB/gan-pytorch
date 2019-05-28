# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
import numpy as np
from PIL import Image
from skimage import img_as_float
import matplotlib.pyplot as plt
from utils import super_resolution as sr


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir, file_list))]
        self.image_filenames = [os.path.join(image_dir, x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = sr.load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                     self.other_dataset)
        else:
            target, input, neigbor = sr.load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                              self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = sr.get_patch(input, target, neigbor, self.patch_size, self.upscale_factor,
                                                  self.nFrames)

        if self.data_augmentation:
            input, target, neigbor, _ = sr.augment(input, target, neigbor)

        flow = [sr.get_flow(input, j) for j in neigbor]

        bicubic = sr.rescale_img(input, self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)

class mod(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def get_loader(config):
    train_size = config.train_image_size
    val_size=config.val_image_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    training_transforms = Compose([sr.RandomGaussianBlur(),
                                      # st.ColorJitter(*[0.1, 0.1, 0.1, 0.1]),
                                      # sr.RandomSizedCrop(size=train_size),
                                      ToTensor(),
                                      # st.Normalize(mean, std)
                                      ])
    val_transforms = Compose([
        # sr.FreeScale(val_size),
                                 ToTensor(),
                                 # st.Normalize(mean, std)
                                 ])

    if config.data_type == 'mod':
        DATASET = mod
    elif config.data_type == 'vimeo90k':
        DATASET = DatasetFromFolder
    else:
        raise NotImplementedError

    training_dataset = DATASET(image_dir=config.image_root,
                               nFrames=config.nFrames,
                               upscale_factor=config.upscale_factor,
                               file_list = config.train_list,
                               patch_size = config.patch_size,
                               other_dataset = config.other_dataset,
                               future_frame = config.future_frame,
                               data_augmentation = config.data_augmentation,
                               transform=training_transforms)
    val_dataset = DATASET(image_dir=config.image_root,
                          nFrames=config.nFrames,
                          upscale_factor=config.upscale_factor,
                          file_list=config.val_list,
                          patch_size=config.patch_size,
                          other_dataset=config.other_dataset,
                          future_frame=config.future_frame,
                          data_augmentation=config.data_augmentation,
                          transform=val_transforms)

    training_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    return {'train': training_dataset, 'val': val_dataset}, \
           {'train': training_loader, 'val': val_loader}


if __name__ == '__main__':
    training_transforms = Compose([sr.ColorJitter(*[0.1, 0.1, 0.1, 0.1]),
                                      sr.RandomSizedCrop(size=256),
                                      sr.RandomGaussianBlur(),
                                      sr.RandomRotate(degree=0),
                                      sr.ToTensor(),
                                      sr.Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))
                                      ])
    test_transforms = Compose([ToTensor(),
                                  sr.Normalize(mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225))
                                  ])
    VocDataset = DatasetFromFolder(image_dir='/home/yhuangcc/data/VOC2012/',
                                   file_list='/home/yhuangcc/data/VOC2012/list/train_aug.txt',
                                   label_file='/home/yhuangcc/ImageSegmentation/datasets/voc/labels',
                                   transform=test_transforms)
    img, mask = VocDataset[0]
    print(img.size())
    print(mask.size())
