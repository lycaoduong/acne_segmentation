from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import os
import torch
import random

class load_data_from_path(Dataset):
    def __init__(self, root_dir, num_class= 2, dataset='train', transform=None, target_transform=None):

        self.data_dir = os.path.join(root_dir, dataset)
        self.img_dir = os.path.join(root_dir, dataset, 'image')
        self.total_img = os.listdir(self.img_dir)

        self.set_name = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.num_class = num_class

    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        data_loader = {'img': image, 'mask': mask}
        data_loader = self.transform(data_loader)
        return data_loader

    def load_image(self, img_idx):
        img_path = os.path.join(self.img_dir, self.total_img[img_idx])
        img = cv2.imread(img_path)
        return img
    def load_mask(self, mask_idx):
        mask = []
        for i in range(self.num_class):
            path = self.data_dir + '/class{}'.format(i+1)
            img_path = os.path.join(path, self.total_img[mask_idx])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask.append(img)
        mask = np.array(mask)
        mask = np.transpose(mask, (1, 2, 0))
        return mask

# root = 'E:/Dataset/Carotid/dataset/Segmentation_B/dataset'
# data = load_data_from_path(root_dir = root)