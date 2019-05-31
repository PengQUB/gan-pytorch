import math
import numbers
import random
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, imgA, imgB):
        if isinstance(imgA, np.ndarray):
            imgA = Image.fromarray(imgA, mode="RGB")
            imgB = Image.fromarray(imgB, mode="RGB")

        assert imgA.size == imgB.size
        for a in self.augmentations:
            imgA, imgB = a(imgA, imgB)

        return imgA, imgB

class ToTensor(object):

    def __call__(self, imgA, imgB):
        return (
            F.to_tensor(imgA),
            F.to_tensor(imgB)
        )

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgA, imgB):
        assert imgA.size == imgB.size
        w, h = imgA.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            imgA.crop((x1, y1, x1 + tw, y1 + th)),
            imgB.crop((x1, y1, x1 + tw, y1 + th))
        )

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgA, imgB):
        assert imgA.size == imgB.size
        w, h = imgA.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return imgA, imgB
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                imgA.resize((ow, oh), Image.BILINEAR),
                imgB.resize((ow, oh), Image.BILINEAR)
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                imgA.resize((ow, oh), Image.BILINEAR),
                imgB.resize((ow, oh), Image.BILINEAR)
            )

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgA, imgB):
        assert imgA.size == imgB.size
        for attempt in range(10):
            area = imgA.size[0] * imgA.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgA.size[0] and h <= imgA.size[1]:
                x1 = random.randint(0, imgA.size[0] - w)
                y1 = random.randint(0, imgA.size[1] - h)

                imgA = imgA.crop((x1, y1, x1 + w, y1 + h))
                imgB = imgB.crop((x1, y1, x1 + w, y1 + h))
                assert imgA.size == (w, h) and imgB.size == (w, h)

                return (
                    imgA.resize((self.size, self.size), Image.BILINEAR),
                    imgB.resize((self.size, self.size), Image.BILINEAR)
                )

        # Fallback
        resize = Resize(self.size)
        crop = CenterCrop(self.size)
        return crop(*resize(imgA, imgB))

class FreeScale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgA, imgB):
        return (
            imgA.resize(self.size, Image.BILINEAR),
            imgB.resize(self.size, Image.BILINEAR)
        )


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgA, imgB):
        return (
            F.normalize(imgA, self.mean, self.std),
            F.normalize(imgB, self.mean, self.std)
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, imgA, imgB):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            F.affine(imgA,
                     translate=(0, 0),
                     scale=1.0,
                     angle=rotate_degree,
                     resample=Image.BILINEAR,
                     fillcolor=(0, 0, 0),
                     shear=0.0),
            F.affine(imgB,
                     translate=(0, 0),
                     scale=1.0,
                     angle=rotate_degree,
                     resample=Image.BILINEAR,
                     fillcolor=(0, 0, 0),
                     shear=0.0)
        )


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgA, imgB):
        if random.random() < self.p:
            return (
                imgA.filter(ImageFilter.GaussianBlur(radius=random.random())),
                imgB.filter(ImageFilter.GaussianBlur(radius=random.random()))
            )
        return imgA, imgB


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, imgA, imgB):
        return (
            transforms.ColorJitter(self.brightness, self.contrast,
                                   self.saturation, self.hue)(imgA),
            transforms.ColorJitter(self.brightness, self.contrast,
                                   self.saturation, self.hue)(imgB)
        )

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, imgA, imgB):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            F.affine(imgA,
                     translate=(0, 0),
                     scale=1.0,
                     angle=rotate_degree,
                     resample=Image.BILINEAR,
                     fillcolor=(0, 0, 0),
                     shear=0.0),
            F.affine(imgB,
                     translate=(0, 0),
                     scale=1.0,
                     angle=rotate_degree,
                     resample=Image.BILINEAR,
                     fillcolor=(0, 0, 0),
                     shear=0.0)
        )

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
