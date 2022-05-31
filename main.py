#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 15, 2022
@author: Justin Xu
"""

import os
import sys
import csv
import time
import random
from google.cloud import bigquery
import numpy as np
import pandas as pd

import torch
import plotly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import VarianceThreshold
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
    np.random.seed(seed_value) # set np random seed
    torch.manual_seed(seed_value) # set torch seed
    random.seed(seed_value) # set python random seed
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # reproducibility
        torch.use_deterministic_algorithms(True)  
        torch.backends.cudnn.benchmark = False

def encode_gender(input):
    if input.lower().replace(" ", "") == "male" or input.lower().replace(" ", "") == "m":
        return 0
    elif input.lower().replace(" ", "") == "female" or input.lower().replace(" ", "") == "f":
        return 1
    else:
        return None

def create_label(mutation, fusion):
    if mutation == 1:
        return 1
    elif fusion == 1:
        return 0
    else:
        return None

def create_label_from_marker(marker):
    if marker == 1:
        return 1
    elif marker == 2:
        return 0
    else:
        return None

def preprocess_sickkids(df, location_2):
    excluded_patient_ids = [9.0, 12.0, 23.0, 37.0, 58.0, 74.0, 78.0, 85.0, 121.0, 122.0, 130.0, 131.0, 138.0, 140.0, 
                             150.0, 171.0, 176.0, 182.0, 204.0, 213.0, 221.0, 224.0, 234.0, 245.0, 246.0, 274.0, 306.0, 
                             311.0, 312.0, 330.0, 334.0, 347.0, 349.0, 352.0, 354.0, 359.0, 364.0, 377.0, 235.0, 243.0, 
                             255.0, 261.0, 264.0, 283.0, 288.0, 293.0, 299.0, 309.0, 325.0, 327.0, 333.0, 334.0, 356.0, 
                             367.0, 376.0, 383.0, 387.0]    

    # define cohort and remove non-patient entries
    code_nanmask = np.isnan(df["code"])
    df = df.loc[~code_nanmask]
    df = df.loc[~df["code"].isin(excluded_patient_ids)]
    df.reset_index(inplace=True, drop=True)

    location_2_OHE = None
    if location_2:
        # one-hot-encoding for Location 2
        location_2_OHE = pd.get_dummies(df["Location_2"], prefix="Location_2")
        df = pd.concat([df, location_2_OHE], axis=1)

    # remove unnecessary columns
    df = df.drop(columns = ['code', 'WT', 'NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                      'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                      'Pathology Coded', 'Location_2', 'Location_Original'])
    
    # create labels and drop non mutation/fusion entries
    df['label'] = df.apply(lambda x: create_label(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
    df = df.drop(columns = ["BRAF V600E final", "BRAF fusion final"])
    label_nanmask = np.isnan(df["label"])
    df = df.loc[~label_nanmask]
    df.reset_index(inplace=True, drop=True)

    # encode gender
    df['Gender'] = df.apply(lambda x: encode_gender(x['Gender']), axis=1)
    df = df.dropna()

    return df, location_2_OHE

def preprocess_stanford(df, df2, location_2, all_location_2_OHEs):
    ### Stanford
    # remove non-patient entries
    code_nanmask = np.isnan(df["code"])
    df = df.loc[~code_nanmask]
    df.reset_index(inplace=True, drop=True)

    if location_2:
        # one-hot-encoding for Location 2
        location_2_OHE = pd.get_dummies(df["Location_2"], prefix="Location_2")
        for i in range(len(location_2_OHE.columns)):
            if location_2_OHE.columns[i] not in all_location_2_OHEs.columns:
                all_location_2_OHEs.insert(i, location_2_OHE.columns[i], 0)
        df = pd.concat([df, location_2_OHE],axis=1)
   
    # remove unnecessary columns
    df = df.drop(columns = ['code', 'FGFR 3','NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                      'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                      'Pathology Coded', 'Location_2', 'Location_Original'])
    
    # create labels and drop non mutation/fusion entries
    df['label'] = df.apply(lambda x: create_label(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
    df = df.drop(columns = ["BRAF V600E final", "BRAF fusion final"])
    label_nanmask = np.isnan(df["label"])
    df = df.loc[~label_nanmask]
    df.reset_index(inplace=True, drop=True)
    
    # encode gender
    df['Gender'] = df.apply(lambda x: encode_gender(x['Gender']), axis=1)
    df = df.dropna()


    ### Stanford_new
    # remove non-patient entries
    df2 = df2.iloc[0:10]

    if include_location_2:
        # one-hot-encoding for Location 2
        location_2_OHE_new = pd.get_dummies(df2["Location_2"], prefix="Location_2")
        for i in range(len(location_2_OHE_new.columns)):
            if location_2_OHE_new.columns[i] not in all_location_2_OHEs.columns:
                all_location_2_OHEs.insert(i, location_2_OHE_new.columns[i], 0)
        df2 = pd.concat([df2, location_2_OHE_new], axis=1)
    
    # remove unnecessary columns
    df2 = df2.drop(columns = ['Code', 'HistoPathologicDiagnosis', 'Location_2'])

    # create labels and drop non relevant marker entries
    df2['label'] = df2.apply(lambda x: create_label_from_marker(x['MolecularMarker']), axis=1)
    df2 = df2.drop(columns = ["MolecularMarker"])
    label_nanmask = np.isnan(df2["label"])
    df2 = df2.loc[~label_nanmask]
    df2.reset_index(inplace=True, drop=True)
    
    # encode gender
    df2['Gender'] = df2.apply(lambda x: encode_gender(x['Gender']), axis=1)
    
    # reformat age
    df2.insert(3, 'Age Dx', df2.apply(lambda x: (x['Age at DGN (months)']/12), axis=1))
    df2 = df2.drop(columns = ["Age at DGN (months)"])

    return df, df2, all_location_2_OHEs

   
# run
if __name__ == '__main__':
    
    random_seed(1, True)
    pd.set_option('display.max_rows', None)

    # Parameters
    num_trials = 1
    k = 1  # number of folds for cross-validation
    n_important_features = 5
    feature_selection_method = "naive"
    include_location_2 = False                   ################# how does this work? why adding SK OHE to Stanford and then concat with df_stanford?

    grid_parameters = {
            'n_estimators': [25, 50, 100],
            'random_state': [42, 100],
            'criterion': ["entropy"],
            'min_samples_leaf': [2, 4, 8],
            'max_depth': [1, 2, 4],
            'max_features': ["auto", None],
            'max_samples': [0.5, 0.75, 1]
        }

    df_sickkids = load_data('Nomogram_study_LGG_data_Nov.27.xlsx', sheet='SK')
    print(f'Rows: {df_sickkids.shape[0]}, Columns: {df_sickkids.shape[1]}')
    df_stanford = load_data('Nomogram_study_LGG_data_Nov.27.xlsx', sheet='Stanford')
    print(f'Rows: {df_stanford.shape[0]}, Columns: {df_stanford.shape[1]}')
    df_stanford_new = load_data('Stanford_new_data_09_21.xlsx')
    print(f'Rows: {df_stanford_new.shape[0]}, Columns: {df_stanford_new.shape[1]}')

    print("Done loading data.\n")

    df_sickkids_processed, all_location_2_OHEs = preprocess_sickkids(df_sickkids, include_location_2)
    print(f'SickKids Data - Rows: {df_sickkids_processed.shape[0]}, Columns: {df_sickkids_processed.shape[1]}')
    print("SickKids data processed.\n")
   
    df_stanford_processed, df_stanford_new_processed, all_location_2_OHEs = preprocess_stanford(df_stanford, df_stanford_new, include_location_2, all_location_2_OHEs)
    print(f'Stanford Data - Rows: {df_stanford_processed.shape[0]}, Columns: {df_stanford_processed.shape[1]}')
    print(f'Stanford_new Data - Rows: {df_stanford_new_processed.shape[0]}, Columns: {df_stanford_new_processed.shape[1]}')

    df_stanford_new_processed.columns = df_stanford_processed.columns
    df_stanford_combined_processed = pd.concat([df_stanford_processed, df_stanford_new_processed], ignore_index=True)
    print(f"Combined Stanford Data - Rows: {df_stanford_combined_processed.shape[0]}, Columns: {df_stanford_combined_processed.shape[1]}")
    print("Stanford data processed.\n")

    print(f"Total number of trial(s): {num_trials}, beginning experiment...")
    execute_experiment(50)
    print(f"{num_trials} trial(s) completed, experiment over.")
