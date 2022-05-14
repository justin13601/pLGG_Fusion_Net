#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 15
@author: Justin Xu
"""

import os
import csv
from google.cloud import bigquery
import numpy as np
import pandas as pd
import plotly
from dash import Dash


def load_data(path, sheet=0):
    filename = os.path.basename(path).strip()
    if isinstance(sheet, str):
        print('Loading ' + filename + ', ' + 'Sheet: ' + sheet + '...')
    else:
        print('Loading ' + filename + '...')
    df_data = pd.read_excel(path, sheet)
    print('Done.')
    return df_data


# run
if __name__ == '__main__':
    df_sickkids = load_data('/home/justinxu/Documents/pLGG/Nomogram_study_LGG_data_Nov.27.xlsx', sheet='SK')
    df_stanford = load_data('/home/justinxu/Documents/pLGG/Nomogram_study_LGG_data_Nov.27.xlsx', sheet='Stanford')
    df_stanford_new = load_data('/home/justinxu/Documents/pLGG/Stanford_new_data_09_21.xlsx')

    print("Finished.")
