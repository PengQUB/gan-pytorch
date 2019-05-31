import math
import numbers
import random
import torch
from torch.autograd import Variable
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F

class ToTensor(object):

    def __call__(self, img):
        return (
            F.to_tensor(img)
        )

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
        )

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR)
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR)
            )

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR)
                )

        # Fallback
        resize = transforms.Resize(self.size)
        crop = transforms.CenterCrop(self.size)
        return crop(resize(img))

class FreeScale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        return (
            img.resize(self.size, Image.BILINEAR)
        )


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (
            F.normalize(img, self.mean, self.std)
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            F.affine(img,
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

    def __call__(self, img):
        if random.random() < self.p:
            return (
                img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            )
        return img


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        return (
            transforms.ColorJitter(self.brightness, self.contrast,
                                   self.saturation, self.hue)(img)
        )

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            F.affine(img,
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
