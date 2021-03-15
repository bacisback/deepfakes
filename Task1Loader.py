#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:00:22 2021

@author: e321075
"""


from matplotlib import pyplot as plt
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import utils



root_dir          = "./Task_1/"
train             = os.path.join(root_dir, "development/")    # train images
test              = os.path.join(root_dir, "evaluation/")    # train images
train_file = os.path.join(root_dir, "train.csv")
test_file  = os.path.join(root_dir, "test.csv")


means=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


class Task1_loader(Dataset):

    def __init__(self, csv_file, phase):
        self.data      = pd.read_csv(csv_file)
        if phase == 'train':
            self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        else:
            self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        input_image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        label = int(label)

            

        # reduce mean
        input_tensor = self.preprocess(input_image)
    
        sample = {'X': input_tensor, 'Y': label}

        return sample

if __name__ == "__main__":
    train_data = Task1_loader(csv_file=train_file, phase='train')

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)