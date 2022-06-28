#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on June 22, 2022
@author: Justin Xu
"""

import os
import sys
import csv
import time
import glob
import socket
import random
import numpy as np
import pandas as pd
from google.cloud import bigquery

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import plotly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from functools import partial


def load_data(path, exclusions, test=False):
    data = {}
    for root, dirs, files in os.walk(path):
        dirs.sort(key=int)
        dirs = list(map(int, dirs))
        dirs = [patient for patient in dirs if patient not in exclusions]
        if test:
            dirs = dirs[:5]
        for d in dirs:
            print(f"Loading Patient {d}...")
            np_filenames = glob.glob(f"{os.path.join(root, f'{d}')}/*/*.npy")
            data[d] = [np.load(np_filenames[0]), np.load(np_filenames[1])]
        break
    return data


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # set np random seed
    torch.manual_seed(seed_value)  # set torch seed
    random.seed(seed_value)  # set python random seed
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # reproducibility
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


# run
if __name__ == '__main__':
    random_seed(1, True)
    pd.set_option('display.max_rows', None)

    # use numpy files instead of .nii
    # no need to normalize images between [0,1] as input images are already preprocessed
    # https://github.com/kenshohara/3D-ResNets-PyTorch

    num_epochs = 100
    num_trials = 20
    batch_size = 8
    learning_rate = 0.01

    begin_time = time.time()

    images = load_data(r'K:\Projects\SickKids_Brain_Preprocessing\preprocessed_FLAIR_from_tumor_seg_dir', exclusions=[], test=True)
