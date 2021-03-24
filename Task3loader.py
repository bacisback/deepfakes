#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:56:28 2021

@author: e321075
"""

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
from face_recognition import FaceRecog
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import cv2
from PIL import Image

root_dir          = "./videos/"



means=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def load_video(filename):
    cap = cv2.VideoCapture(filename)
    ret = True
    i = 0
    j = 0
    out = []
    while(cap.isOpened()):
        if j == 10:
            break
        try:
            ret, frame_in = cap.read()
        except:
            continue
        if ret == False:
            break
        out.append(Image.fromarray(frame_in))
        if i%3==0:
            j +=1
        i += 1
    return out
class Task3_loader(Dataset):

    def __init__(self, dir_vids=root_dir, phase='train'):
        self.data      = [os.path.join(dir_vids,name) for name in os.listdir(dir_vids) if "json" not in name]
        self.imgs = []
        self.labels = {}
        self.video=-1
        meta =  pd.read_json(dir_vids+"metadata.json")
        for name in os.listdir(dir_vids):
            if "json" in name:
                continue
            
            self.labels[os.path.join(dir_vids,name)] = int(meta[name]["label"] != "FAKE")
        print(self.labels)
        if phase == 'train':
            self.preprocess = [
                            FaceRecog(margin=7),
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
        else:
            self.preprocess = [
                            FaceRecog(margin=7),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
        self.preprocess = transforms.Compose(self.preprocess)

    def __len__(self):
        return len(self.data)*10

    def __getitem__(self, idx):
        video = idx//10
        frame = video%10
        if video != self.video:
            self.video = video
            self.imgs = load_video(self.data[video])
        try:
            input_image   = self.imgs[frame*3]
        except:
            print(frame)
            input_image   = self.imgs[frame]
        try:
            label = self.labels[self.data[video]]
        except:
            print(self.data[video])
            

        # reduce mean
        input_tensor = self.preprocess(input_image)
    
        sample = {'X': input_tensor, 'Y': label}

        return sample

if __name__ == "__main__":
    train_data = Task3_loader()

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'])

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
