# -*- coding: utf-8 -*-
import os
import glob
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from utils import cyclegan as cyc


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, unaligned=False, transform=None, mode='train'):
        super(DatasetFromFolder, self).__init__()

        self.files_A = sorted(glob.glob(os.path.join(image_dir, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(image_dir, '%sB' % mode) + '/*.*'))

        self.unaligned = unaligned
        self.transform = transform

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B = Image.open(self.files_B[index % len(self.files_B)])

        return self.transform(img_A, img_B)

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

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
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    training_transforms = cyc.Compose([cyc.FreeScale(train_size),
                                       cyc.ToTensor(),
                                       cyc.Normalize(mean=mean, std=std),
                                       ])
    val_transforms = cyc.Compose([cyc.FreeScale(val_size),
                                  cyc.ToTensor(),
                                  cyc.Normalize(mean=mean, std=std),
                                  ])

    DATASET = DatasetFromFolder

    training_dataset = DATASET(image_dir=config.image_root,
                               unaligned=config.unaligned,
                               transform=training_transforms)
    val_dataset = DATASET(image_dir=config.image_root,
                          unaligned=config.unaligned,
                          transform=val_transforms)

    training_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    return {'train': training_dataset, 'val': val_dataset}, \
           {'train': training_loader, 'val': val_loader}


if __name__ == '__main__':
    training_transforms = Compose([cyc.RandomSizedCrop(size=256),
                                      cyc.RandomRotate(degree=0),
                                      cyc.ToTensor(),
                                      cyc.Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))
                                      ])
    test_transforms = Compose([ToTensor(),
                                  cyc.Normalize(mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225))
                                  ])
    VocDataset = DatasetFromFolder(image_dir='/home/test/',
                                   transform=test_transforms)
    img, mask = VocDataset[0]
    print(img.size())
    print(mask.size())
