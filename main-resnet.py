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
import random
import numpy as np
import pandas as pd
from google.cloud import bigquery

import torch
import plotly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_data(path, sheet=0):
    filename = os.path.basename(path).strip()
    if isinstance(sheet, str):
        print(f'Loading {filename}, Sheet: {sheet}...')
    else:
        print('Loading ' + filename + '...')
    df_data = pd.read_excel(path, sheet)
    return df_data


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