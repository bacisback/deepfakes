
from csv_create import create_csv
from BinaryClassifier import BinaryClassifier
from Task1Loader import Task1_loader
from runner import *

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models

def main():

    create_csv()

    batch_size = 5

    model = models.resnet18(pretrained=True)
    clssf = BinaryClassifier(model)

    train_data = Task1_loader("./Task_1/train.csv", phase='train')
    test_data = Task1_loader("./Task_1/test.csv", phase='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8)


    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train(clssf, train_loader, valid_loader, criterion, optimizer, 10, device='cpu')


if __name__ == '__main__':
    main()
