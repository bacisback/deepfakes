from Task1Loader import Task1_loader
from runner import *
from face_recognition import FaceRecog

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model
import torchvision.models as models

import os

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()
    self.l = nn.Linear(in_f, 512)
    self.d = nn.Dropout(0.75)
    self.o = nn.Linear(512, out_f)
    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)
    self.r = nn.ReLU()

  def forward(self, x):
    x = self.f(x)
    x = self.d(x)

    x = self.l(x)
    x = self.r(x)
    x = self.d(x)

    out = self.o(x)
    return out

class FCN(torch.nn.Module):
  def __init__(self, base, in_f):
    super(FCN, self).__init__()
    self.base = base
    self.h1 = Head(in_f, 1)
    self.classif = nn.Sigmoid()
  
  def forward(self, x):
    x = self.base(x)
    x = self.h1(x)
    return x, self.classif(x)


def main():

    batch_size = 55

    # model = models.resnet18(pretrained=False)
    model = get_model("xception", pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) # xcep
    model = FCN(model, 2048)

    train_data = Task1_loader("./Task_1/train.csv", phase='train')
    test_data = Task1_loader("./Task_1/test.csv", phase='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0018, momentum=0.27)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train(model, train_loader, valid_loader, criterion, optimizer, 10, device='cpu')

if __name__ == '__main__':
    main()