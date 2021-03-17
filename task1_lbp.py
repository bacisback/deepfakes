#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:09:42 2021

@author: e321075
"""

import numpy as np
from skimage import feature
import os
import pandas as pd
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class LBPExtractor:
  
  def __init__(self, points, radius):
    self.points = points
    self.radius = radius
  
  def get_image_hist(self, image, eps=1e-7):
    sigma_min = 1
    sigma_max = 16
    hist = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    
    hist = hist.astype("float")
    
    return hist

def get_hist_from_img_route(image_route, lbp_extractor):
  image = cv2.imread(image_route)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return lbp_extractor.get_image_hist(gray_image)


train_data = "./Task_1/train.csv"
test_data = "./Task_1/test.csv"

lbp_extractor = LBPExtractor(points=24, radius=15)
train_loader =  pd.read_csv(train_data)
train = []
labels = []
for idx in range(len(train_loader)):
    train.append(get_hist_from_img_route(train_loader.iloc[idx, 0], lbp_extractor))
    labels.append(train_loader.iloc[idx, 1])
    
df = pd.DataFrame(train).assign(label=labels)
df.describe()
df.to_csv('lbp_train.csv')
df = pd.read_csv('lbp_train.csv')
y = df[['label']].values.ravel()
X = df.drop(['label', 'Unnamed: 0'], axis=1)

train_loader =  pd.read_csv(test_data)
train = []
labels = []
for idx in range(len(train_loader)):
    train.append(get_hist_from_img_route(train_loader.iloc[idx, 0], lbp_extractor))
    labels.append(train_loader.iloc[idx, 1])
    
df = pd.DataFrame(train).assign(label=labels)
df.to_csv('lbp_val.csv')
df = pd.read_csv('lbp_val.csv')

y_test = df[['label']].values.ravel()
X_test = df.drop(['label', 'Unnamed: 0'], axis=1)

gnb = GaussianNB()
model = gnb.fit(X, y)

y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))

model = RandomForestClassifier(n_estimators=3000, max_depth=5, random_state=0).fit(X, y)

y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))