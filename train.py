#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:47:19 2021

@author: e321075
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from Cityscapes_loader import CityscapesDataset
#from CamVid_loader import CamVidDataset
from Task1Loader import Task1_loader

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from torchvision import utils


model_dir = "./models"
model_path = os.path.join(model_dir, "scratch_resnext")

model = torch.load(model_path)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

batch_size = 16
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5


train_data = Task1_loader("./Task_1/train.csv", phase='train')
test_data = Task1_loader("./Task_1/test.csv", phase='test')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader   = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8)
activation  = nn.Sigmoid()
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a

def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = model(inputs)
            outputs = activation(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(model, model_path+"train")

        val(epoch)



def val(epoch):
    model.eval()
    tot_loss = 0
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs = Variable(batch['X'])
            labels = Variable(batch['Y'])
        
        
        output = model(inputs)
        output = activation(output)
        labels = labels.type(torch.float)
        output = output.type(torch.float).squeeze()
        #print(output, labels)
        loss = criterion(output, labels)
        tot_loss += loss
        print(loss)
    print(tot_loss)
    

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
        
        
        
        