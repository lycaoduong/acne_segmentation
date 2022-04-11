import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

class Normalizer(object):

    def __init__(self, mean=[0.1, 0.1, 0.1], std=[0.2, 0.2, 0.2]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, mask  = sample['img'], sample['mask']
        # {'img': image, 'annot': annot, 'filename': filename}
        # return {'img': ((image.astype(np.float32) / 255.0 - self.mean) / self.std), 'mask': mask.astype(np.float32) / 255.0}
        return {'img': (image.astype(np.float32) / 255.0),
                'mask': mask.astype(np.float32) / 255.0}

def rotate_duong(image, mask):
    """
    :param image:
    :param mask:
    :return: Sample after rotation
    """
    sample = {'img': image, 'mask': mask}
    return sample

def scale_duong(image, mask , scale_range=[1.1, 1.5, 0.1]):
    """
    :param image:
    :param mask:
    :param scale_range:
    :return:
    """
    sample = {'img': image, 'mask': mask}
    return sample

def flipx_duong(image, mask):
    """
    :param image:
    :param mask:
    :param scale_range:
    :return:
    """
    sample = {'img': image, 'mask': mask}
    return sample

def flipy_duong(image, mask):
    """
    :param image:
    :param mask:
    :param scale_range:
    :return:
    """
    sample = {'img': image, 'mask': mask}
    return sample

def enhace_duong(image, mask):
    """
    :param image:
    :param mask:
    :param scale_range:
    :return:
    """
    sample = {'img': image, 'mask': mask}
    return sample

def sliding(image, mask):
    """
    :param image:
    :param mask:
    :param scale_range:
    :return:
    """
    sample = {'img': image, 'mask': mask}
    return sample

class augmentation_duong(object):

    def __init__(self,
                 rotate=0.5,
                 scale=0.5,
                 scale_range=[1.1, 1.5, 0.2],
                 flipx=0.5,
                 flipy=0.5,
                 enhance_contrast=0.5,
                 sliding=0.5
                 ):
        self.rotate = rotate
        self.scale = scale
        self.scale_range = scale_range
        self.flipx = flipx
        self.flipy = flipy
        self.enhance_contrast = enhance_contrast
        self.sliding = sliding

    def __call__(self, sample):
        image, mask  = sample['img'], sample['mask']
        if self.rotate < np.random.rand():
            sample = rotate_duong(image, mask)
        if self.scale < np.random.rand():
            sample = scale_duong(image, mask, self.scale_range)
        if self.flipx < np.random.rand():
            sample = flipx_duong(image, mask)
        if self.flipy < np.random.rand():
            sample = flipy_duong(image, mask)
        if self.enhance_contrast < np.random.rand():
            sample = enhace_duong(image, mask)
        if self.sliding < np.random.rand():
            sample = sliding(image, mask)
        return sample

class Resizer(object):
    def __init__(self, img_size=512, use_offset=True, mean=48):
        self.img_size = img_size
        self.use_offset = use_offset
        self.mean = np.array(mean)

    def __call__(self, sample):
        image, mask = sample['img'], sample['mask']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # print(image.shape, mask.shape)
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)


        # new_image = np.ones((self.img_size, self.img_size, 3)) * self.mean
        new_image = np.zeros((self.img_size, self.img_size, image.shape[2]))
        if mask.shape == 2:
            new_mask = np.zeros((self.img_size, self.img_size))
        else:
            new_mask = np.zeros((self.img_size, self.img_size, mask.shape[2]))

        if self.use_offset:
            offset_w = (self.img_size - resized_width) // 2
            offset_h = (self.img_size - resized_height) // 2
            new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image
            new_mask[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = mask

        else:
            new_image[0:resized_height, 0:resized_width] = image
            new_mask[0:resized_height, 0:resized_width] = mask
            # return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots),
            #         'scale': scale}

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'mask': torch.from_numpy(new_mask).to(torch.long)}

# class patch_extract(object):