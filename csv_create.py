#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:46:15 2021

@author: e321075
"""

from PIL import Image
import os
from collections import namedtuple
import re


#############################
	# global variables #
#############################
root_dir          = "./Task_2_3/"
train             = os.path.join(root_dir, "development/")    # train images
test              = os.path.join(root_dir, "evaluation/")    # train images


def create_csv():
    train_file = os.path.join(root_dir, "train.csv")
    test_file = os.path.join(root_dir, "test.csv")
    train_file = open(train_file, "w")
    train_file.write("img,label\n")
    test_file = open(test_file, "w")
    test_file.write("img,label\n")
	
    for int_label, label in enumerate(["fake/", "real/"]):
        folder = os.path.join(train, label)
        for name in os.listdir(folder):
            video_name = os.path.join(folder, name)
            for filename in os.listdir(video_name):
                filename_img = os.path.join(video_name, filename)
                train_file.write("{},{}\n".format(filename_img, int_label))
        folder = os.path.join(test, label)
        for name in os.listdir(folder):
            video_name = os.path.join(folder, name)
            for filename in os.listdir(video_name):
                filename_img = os.path.join(video_name, filename)
                test_file.write("{},{}\n".format(filename_img, int_label))
    train_file.close()
    test_file.close()


if __name__ == '__main__':
	create_csv()
