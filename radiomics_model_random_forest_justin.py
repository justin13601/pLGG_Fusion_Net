#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 15

@author: Kareem
"""
import numpy as np
import pandas as pd
import os
from os.path import dirname as up
import torch
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time
import socket
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Seeding#####################################################################
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(17, True)
##############################################################################


if __name__ == "__main__":
    # Experiment metrics
    k = 5  # Number of folds for cross-validation
    n_important_features = 50
    feature_selection_method = "naive"
    include_location_2 = True

    ############################################Data Preprocessing######################################################
    # Pointing the excel file which contain the data (SK and Stanford)
    if socket.gethostname()=="kareem-XPS-13-9310":
        root_dir = up(up(up(__file__)))
        data_path = os.path.join(root_dir, "Data", "Nomogram", "Nomogram_study_LGG_data_Nov.27.xlsx")
        data_SK_input = pd.read_excel(data_path, sheet_name="SK")
        data_Stanford_input = pd.read_excel(data_path, sheet_name="Stanford")
        data_Stanford_input_new = pd.read_excel(os.path.join(root_dir, "Data", "Nomogram", "Stanford_new_data_09_21 (1).xlsx"))
    else:
        data_SK_input = pd.read_excel(
            "/media/Datasets/MedicalImages/BrainData//SickKids//Radiomics/Nomogram/Nomogram_study_LGG_data_Nov.27.xlsx",
            sheet_name="SK", engine="openpyxl")
        data_Stanford_input = pd.read_excel(
            "/media/Datasets/MedicalImages/BrainData//SickKids//Radiomics/Nomogram/Nomogram_study_LGG_data_Nov.27.xlsx",
            sheet_name="Stanford", engine="openpyxl")
        data_Stanford_input_new = pd.read_excel(
            os.path.join("/media/Datasets/MedicalImages/BrainData//SickKids//Radiomics/Nomogram/Stanford_new_data_09_21.xlsx"), engine='openpyxl')

    num_trials = 10

    param_grid = {
            'n_estimators': [25, 50, 100],
            'random_state': [42, 100],
            'criterion': ["entropy"],
            'min_samples_leaf': [2, 4, 8],
            'max_depth': [1, 2, 4],
            'max_features': ["auto", None],
            'max_samples': [0.5, 0.75, 1]
        }


    #############################################Preprocessing SK#######################################################
    # Removing the last rows (which do not belong to examples)
    nanmask = np.isnan(data_SK_input["code"])
    data_SK = data_SK_input[~nanmask]
    data_SK = data_SK.reindex()

    # Remove patients that are to be excluded from the study (the ones with missing data will be dropped later)
    excluded_patients = [9.0, 12.0, 23.0, 37.0, 58.0, 74.0, 78.0, 85.0, 121.0, 122.0, 130.0, 131.0, 138.0, 140.0, 150.0,
                         171.0, 176.0, 182.0, 204.0, 213.0, 221.0, 224.0, 234.0, 245.0, 246.0, 274.0, 306.0, 311.0,
                         312.0, 330.0, 334.0, 347.0, 349.0, 352.0, 354.0, 359.0, 364.0, 377.0,
                         # Above are all new additions, previously was assuming they would be caught due to missing data
                         235.0, 243.0, 255.0, 261.0, 264.0, 283.0, 288.0, 293.0,
                              299.0, 309.0, 325.0, 327.0, 333.0, 334.0, 356.0, 367.0,
                              376.0, 383.0, 387.0]
    data_SK = data_SK[~data_SK["code"].isin(excluded_patients)]
    data_SK = data_SK.reindex()

    if include_location_2:
        # Add one-hot-encoding for Location 2
        location_2_one_hot_SK = pd.get_dummies(data_SK["Location_2"], prefix="Location_2")
        data_SK = pd.concat([data_SK,location_2_one_hot_SK],axis=1)

    # Remove data that we don't need for this analysis
    data_SK = data_SK.drop(columns = ['code', 'WT', 'NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                      'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                      'Pathology Coded', 'Location_2', 'Location_Original'])

    # Create label column
    def f(mut, fus):
        if mut == 1:
            return 1
        elif fus ==1:
            return 0
        else:
            return None
    data_SK['label'] = data_SK.apply(lambda x: f(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
    data_SK = data_SK.drop(columns = ["BRAF V600E final", "BRAF fusion final"])

    # Drop rows where the outcome is not mutation or fusion
    nanmask = np.isnan(data_SK["label"])
    data_SK = data_SK[~nanmask]
    data_SK = data_SK.reindex()

    # Convert male/female to binary
    def gender_binary(gen):
        if gen == "Male":
            return 0
        elif gen == "Male ":
            return 0
        elif gen == "Female":
            return 1
        elif gen == "female":
            return 1
        else:
            return None

    data_SK['Gender'] = data_SK.apply(lambda x: gender_binary(x['Gender']), axis=1)

    # Remove all remaining rows that are missing data (marked in red on the spread sheet)
    data_SK = data_SK.dropna()

    ############################################Preprocessing Stanford##################################################
    # Removing the last rows (which do not belong to examples)
    nanmask = np.isnan(data_Stanford_input["code"])
    data_Stanford = data_Stanford_input[~nanmask]
    data_Stanford = data_Stanford.reindex()

    if include_location_2:
        # Add one-hot-encoding for Location 2
        location_2_one_hot_Stanford = pd.get_dummies(data_Stanford["Location_2"], prefix="Location_2")
        for i in range(len(location_2_one_hot_SK.columns)):
            if location_2_one_hot_SK.columns[i] not in location_2_one_hot_Stanford.columns:
                location_2_one_hot_Stanford.insert(i,location_2_one_hot_SK.columns[i],0)
        data_Stanford = pd.concat([data_Stanford,location_2_one_hot_Stanford],axis=1)

    # Remove data that we don't need for this analysis
    data_Stanford = data_Stanford.drop(columns = ['code', 'FGFR 3','NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                      'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                      'Pathology Coded', 'Location_2', 'Location_Original'])


    # Create label column
    def f(mut, fus):
        if mut == 1:
            return 1
        elif fus ==1:
            return 0
        else:
            return None
    data_Stanford['label'] = data_Stanford.apply(lambda x: f(x['BRAF V600E final'], x['BRAF fusion final']), axis=1)
    data_Stanford = data_Stanford.drop(columns = ["BRAF V600E final", "BRAF fusion final"])

    # Drop rows where the outcome is not mutation or fusion
    nanmask = np.isnan(data_Stanford["label"])
    data_Stanford = data_Stanford[~nanmask]
    data_Stanford = data_Stanford.reindex()

    # Convert male/female to binary
    def gender_binary(gen):
        if gen == "Male":
            return 0
        elif gen == "Male ":
            return 0
        elif gen == "Female":
            return 1
        else:
            return None

    data_Stanford['Gender'] = data_Stanford.apply(lambda x: gender_binary(x['Gender']), axis=1)

    # Remove all remaining rows that are missing data (marked in red on the spread sheet)
    data_Stanford = data_Stanford.dropna()

    ############################################Preprocessing New Stanford##############################################
    # Removing the last rows (which do not belong to examples)
    data_Stanford_new = data_Stanford_input_new.iloc[0:10]

    if include_location_2:
        # Add one-hot-encoding for Location 2
        location_2_one_hot_Stanford = pd.get_dummies(data_Stanford_new["Location_2"], prefix="Location_2")
        for i in range(len(location_2_one_hot_SK.columns)):
            if location_2_one_hot_SK.columns[i] not in location_2_one_hot_Stanford.columns:
                location_2_one_hot_Stanford.insert(i,location_2_one_hot_SK.columns[i],0)
        data_Stanford_new = pd.concat([data_Stanford_new,location_2_one_hot_Stanford],axis=1)

    # Remove data that we don't need for this analysis
    data_Stanford_new = data_Stanford_new.drop(columns = ['Code', 'HistoPathologicDiagnosis', 'Location_2'])


    # Create label column
    def f(marker):
        if marker == 1:
            return 1
        elif marker == 2:
            return 0
        else:
            return None
    data_Stanford_new['label'] = data_Stanford_new.apply(lambda x: f(x['MolecularMarker']), axis=1)
    data_Stanford_new = data_Stanford_new.drop(columns = ["MolecularMarker"])

    # Drop rows where the outcome is not mutation or fusion
    nanmask = np.isnan(data_Stanford_new["label"])
    data_Stanford_new = data_Stanford_new[~nanmask]
    data_Stanford_new = data_Stanford_new.reindex()

    # Convert male/female to binary
    def gender_binary(gen):
        if gen == "M":
            return 0
        elif gen == "F":
            return 1
        else:
            return None
    data_Stanford_new['Gender'] = data_Stanford_new.apply(lambda x: gender_binary(x['Gender']), axis=1)

    # Convert age from months to years
    def months_to_years(age):
            return age/12
    # data_Stanford_new['Age Dx'] = data_Stanford_new.apply(lambda x: months_to_years(x['Age at DGN (months)']), axis=1)
    data_Stanford_new.insert(3, 'Age Dx', data_Stanford_new.apply(lambda x: months_to_years(x['Age at DGN (months)']), axis=1))
    data_Stanford_new = data_Stanford_new.drop(columns = ["Age at DGN (months)"])

    ####################################Print out summary of preprocessing steps########################################
    print(f"SK data started with {data_SK_input.shape[0]} rows, post preprocessing we have"
          f" {data_SK.shape[0]} have rows")
    print(f"Stanford data started with {data_Stanford_input.shape[0]} rows, post preprocessing we have"
          f" {data_Stanford.shape[0]} have rows")
    print(f"New Stanford data started with {data_Stanford_input_new.shape[0]} rows, post preprocessing we have"
          f" {data_Stanford_new.shape[0]} have rows")
    data_Stanford_new.columns = data_Stanford.columns
    data_Stanford = pd.concat([data_Stanford,data_Stanford_new], ignore_index=True)
    print(f"Combined Stanford dataset size:"
          f" {data_Stanford.shape[0]}")
    print(f"Num trials: {num_trials}")

    ###################################################Experiments######################################################

    # The number of times you want to go through the entire thing, starting from test/train split
    for t in range(num_trials):
        print(f"Trial: {t}")
        se = np.random.randint(10000) # New seed for splitting data each trial
        result = []
        naive_combo_auc = []

        # Data splitter that we will use for buildinf all the different models
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=se)

        # Separate data into x and y
        # SK
        Y = data_SK["label"].to_numpy()
        X = data_SK.drop(columns = ["label"])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=se)

        # Stanford
        Y_stanford = data_Stanford["label"].to_numpy()
        X_stanford = data_Stanford.drop(columns=["label"])

        # Create df without clinical data
        clinical_vars = ['Location_1', 'Gender', 'Age Dx']
        if include_location_2:
            clinical_vars += list(location_2_one_hot_SK.columns)
        X_train_no_clinical = X_train.drop(clinical_vars, axis =1)
        X_test_no_clinical = X_test.drop(clinical_vars, axis =1)
        X_stanford_no_clinical = X_stanford.drop(clinical_vars, axis =1)


        # Drop columns which have a high correlation to another column

        # Radiomics only
        cor_matrix = X_train_no_clinical.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]
        X_train_no_clinical = X_train_no_clinical.drop(to_drop, axis=1)
        X_test_no_clinical = X_test_no_clinical.drop(to_drop, axis=1)
        X_stanford_no_clinical = X_stanford_no_clinical.drop(to_drop, axis=1)

        # Apply variance thresholding
        # Radiomics only
        selector = VarianceThreshold()
        X_train_no_clinical = selector.fit_transform(X_train_no_clinical)
        X_test_no_clinical = selector.transform(X_test_no_clinical)
        X_stanford_no_clinical = selector.transform(X_stanford_no_clinical)

        # Fit random forest on all radiomics features
        rfc_cv = RandomForestClassifier(n_jobs=-1)
        CV_rfc = GridSearchCV(estimator=rfc_cv, param_grid=param_grid, cv=kf, verbose = 0, scoring="roc_auc",n_jobs=-1,
                              return_train_score=True)
        temp = CV_rfc.fit(X_train_no_clinical, Y_train)
        best_model_all_radiomics = temp.best_estimator_
        best_score_all_radiomics = CV_rfc.best_score_
        

