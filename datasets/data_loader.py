# -*- coding: utf-8 -*-
import os
import glob
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from utils import cyclegan as cyc
from multiprocessing import Lock, Process, Queue, current_process
import multiprocessing
import time
import queue  # imported for using queue.Empty exception


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, unaligned=False, transform=None, mode='val'):
        super(DatasetFromFolder, self).__init__()

        self.imgA_list = multiprocessing.Manager().list(range(24646))
        self.imgB_list = multiprocessing.Manager().list(range(24646))

        self.unaligned = unaligned
        self.transform = transform

        self.image_dir = image_dir
        self.mode = mode

    def __getitem__(self, index):
        self.main_test()
        return self.transform(self.imgA_list[index], self.imgB_list[index])
        # return self.imgA_list[index], self.imgB_list[index]

    def __len__(self):
        return max(len(self.imgA_list), len(self.imgB_list))

    def do_job(self, tasks_to_accomplish):

        while True:
            try:
                img_path = tasks_to_accomplish.get_nowait()
                for i in range(len(img_path[0])):
                    self.imgA_list[i] = Image.open(img_path[0][i])
                    self.imgB_list[i] = Image.open(img_path[1][i])

            except queue.Empty:

                break

        return True

    def main_test(self):

        imgA_path = sorted(glob.glob(self.image_dir + '%sA' % self.mode + '/*.*'))
        imgB_path = sorted(glob.glob(self.image_dir + '%sB' % self.mode + '/*.*'))
        number_of_task = len(imgA_path)
        number_of_processes = 16
        tasks_to_accomplish = Queue()
        # tasks_that_are_done = Queue()
        processes = []
        imgA_path = sorted(glob.glob(self.image_dir + '%sA' % self.mode + '/*.*'))
        imgB_path = sorted(glob.glob(self.image_dir + '%sB' % self.mode + '/*.*'))
        img_path = [imgA_path, imgB_path]  # img_path[0], img_path[1]

        for i in range(number_of_task):
            tasks_to_accomplish.put(img_path)

        # creating processes
        for w in range(number_of_processes):
            p = Process(target=self.do_job, args=(tasks_to_accomplish,))  # creat process class
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()


        return True


class mod(Dataset):  # 有什么用？
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def get_loader(config):
    train_size = config.train_image_size
    val_size = config.val_image_size
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
    test_transforms = cyc.Compose([cyc.ToTensor(),
                                   cyc.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
                                   ])

    VocDataset = DatasetFromFolder(image_dir='/Users/momo/Desktop/multipro_data/',
                                   transform=test_transforms)
    img, mask = VocDataset[0]
    # img, mask = test_transforms(img, mask)
    print(img)

