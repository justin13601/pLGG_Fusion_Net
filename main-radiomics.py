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
        df = pd.concat([df, location_2_OHE], axis='columns')

    # remove unnecessary columns
    df = df.drop(columns=['code', 'WT', 'NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                          'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                          'Pathology Coded', 'Location_2', 'Location_Original'])

    # create labels and drop non mutation/fusion entries
    df['label'] = df.apply(lambda x: create_label(x['BRAF V600E final'], x['BRAF fusion final']), axis='columns')
    df = df.drop(columns=["BRAF V600E final", "BRAF fusion final"])
    label_nanmask = np.isnan(df["label"])
    df = df.loc[~label_nanmask]
    df.reset_index(inplace=True, drop=True)

    # encode gender
    df['Gender'] = df.apply(lambda x: encode_gender(x['Gender']), axis='columns')
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
        df = pd.concat([df, location_2_OHE], axis='columns')

    # remove unnecessary columns
    df = df.drop(columns=['code', 'FGFR 3', 'NF1', 'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1',
                          'FGFR 2', 'FGFR 4', 'Further gen info', 'Notes', 'Pathology Dx_Original',
                          'Pathology Coded', 'Location_2', 'Location_Original'])

    # create labels and drop non mutation/fusion entries
    df['label'] = df.apply(lambda x: create_label(x['BRAF V600E final'], x['BRAF fusion final']), axis='columns')
    df = df.drop(columns=["BRAF V600E final", "BRAF fusion final"])
    label_nanmask = np.isnan(df["label"])
    df = df.loc[~label_nanmask]
    df.reset_index(inplace=True, drop=True)

    # encode gender
    df['Gender'] = df.apply(lambda x: encode_gender(x['Gender']), axis='columns')
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
        df2 = pd.concat([df2, location_2_OHE_new], axis='columns')

    # remove unnecessary columns
    df2 = df2.drop(columns=['Code', 'HistoPathologicDiagnosis', 'Location_2'])

    # create labels and drop non relevant marker entries
    df2['label'] = df2.apply(lambda x: create_label_from_marker(x['MolecularMarker']), axis='columns')
    df2 = df2.drop(columns=["MolecularMarker"])
    label_nanmask = np.isnan(df2["label"])
    df2 = df2.loc[~label_nanmask]
    df2.reset_index(inplace=True, drop=True)

    # encode gender
    df2['Gender'] = df2.apply(lambda x: encode_gender(x['Gender']), axis='columns')

    # reformat age
    df2.insert(3, 'Age Dx', df2.apply(lambda x: (x['Age at DGN (months)'] / 12), axis='columns'))
    df2 = df2.drop(columns=["Age at DGN (months)"])

    return df, df2, all_location_2_OHEs


def split_data(data, seed):
    Y = data["label"].to_numpy()
    X = data.drop(columns=["label"])

    # train-validation split, 25% validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)
    return X_train, X_val, Y_train, Y_val


def feature_selection(method, X_train):
    if method == "naive":
        print('naive')
    if method == "rfe":
        print('rfe')
    if method == "other":
        print('other')
    return X_train


def remove_correlated_features(X_train=None, X_val=None, X_test=None, threshold=0.98):
    if X_train is not None:
        correlation_matrix = X_train.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        X_train = X_train.drop(to_drop, axis='columns')
        if X_val is not None:
            X_val = X_val.drop(to_drop, axis='columns')
        if X_test is not None:
            X_test = X_test.drop(to_drop, axis='columns')
    return X_train, X_val, X_test


def variance_threshold(X_train=None, X_val=None, X_test=None):
    selector = VarianceThreshold()
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    return X_train, X_val, X_test


def execute_experiment(num_trials, k, grid_parameters, df_SK, df_SF, OHEs, location_2, feature_selection_method):
    df_result = pd.DataFrame(columns=['Trial #', 'Training AUC', 'Validation AUC', 'SF Testing AUC'])
    for t in range(num_trials):
        print("\r", f"Trial: {t + 1}/{num_trials}...", end="")

        # set new seed for each trial
        seed = np.random.randint(10000)
        result = []
        naive_combo_auc = []

        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        X_train, X_val, Y_train, Y_val = split_data(df_SK, seed)  # sickkids internal train/val

        # stanford external test
        Y_test = df_SF["label"].to_numpy()
        X_test = df_SF.drop(columns=["label"])

        # remove clinical data columns
        clinical_vars = ['Location_1', 'Gender', 'Age Dx']
        if location_2:
            clinical_vars += list(OHEs.columns)
        X_train = X_train.drop(clinical_vars, axis='columns')
        X_val = X_val.drop(clinical_vars, axis='columns')
        X_test = X_test.drop(clinical_vars, axis='columns')

        # remove highly correlated features and apply variance threshold
        X_train, X_val, X_test = remove_correlated_features(X_train, X_val, X_test, threshold=0.98)
        X_train, X_val, X_test = variance_threshold(X_train, X_val, X_test)

        # fit radiomics model on training data
        rfc = RandomForestClassifier(n_jobs=-1)  # all CPUs used
        rfc_cv = GridSearchCV(estimator=rfc, param_grid=grid_parameters, cv=kfold, verbose=0, scoring="roc_auc",
                              n_jobs=-1, return_train_score=True, refit=True)
        fit_model = rfc_cv.fit(X_train, Y_train)
        best_model_radiomics = fit_model.best_estimator_
        best_score_radiomics = rfc_cv.best_score_

        # get predictions for radiomics model
        predictions_val_radiomics = best_model_radiomics.predict_proba(X_val)
        predictions_test_radiomics = best_model_radiomics.predict_proba(X_test)

        training_auc = fit_model.cv_results_['mean_train_score'][fit_model.best_index_]
        validation_auc = roc_auc_score(Y_val, predictions_val_radiomics[:, 1])
        testing_auc = roc_auc_score(Y_test, predictions_test_radiomics[:, 1])

        print(f"\nTraining AUC: {training_auc}")
        print(f"Validation AUC: {validation_auc}")
        print(f"Stanford Testing AUC: {testing_auc}")

        # # store validation scores for radiomics and feature selection model
        # all_folds_true_val = []
        # all_folds_preds_val_radiomics = []
        # all_folds_preds_val_feature_selection = []
        #
        # # feature selection model
        # X_train_feature_selection = X_train
        # if feature_selection_method is not None:
        #     X_train_feature_selection = feature_selection(feature_selection_method, X_train)
        #
        #     rfc_feature_selection = RandomForestClassifier(n_jobs=-1)
        #     rfc_feature_selection_CV = GridSearchCV(estimator=rfc_feature_selection, param_grid=grid_parameters, cv=kfold, verbose=0, scoring="roc_auc", n_jobs=-1, return_train_score=True)
        #     fit_model_feature_selection = rfc_feature_selection_CV.fit(X_train_feature_selection, Y_train)
        #     best_model_feature_selection = fit_model_feature_selection.best_estimator_
        #     best_score_feature_selection = rfc_feature_selection_CV.best_score_
        #
        #     predictions_val_feature_selection = best_model_feature_selection.predict_proba(X_val)
        #     predictions_test_feature_selection = best_model_feature_selection.predict_proba(X_test)
        #
        # kfold2 = KFold(n_splits=X_train.shape[0], shuffle=True, random_state=seed)
        #
        # for train_index, val_index in kfold2.split(X_train, Y_train):  # for each fold
        #     temp1 = best_model_radiomics.fit(X_train[train_index], Y_train[train_index])
        #     all_folds_preds_val_radiomics.extend(temp1.predict_proba(X_train[val_index])[:,1])
        #     if feature_selection_method is not None:
        #         temp2 = best_model_feature_selection.fit(X_train_feature_selection[train_index], Y_train[train_index])
        #         all_folds_preds_val_feature_selection.extend(temp2.predict_proba(X_train_feature_selection[val_index])[:, 1])
        #     all_folds_true_val.extend(Y_train[val_index])
        #
        # best_model_radiomics = best_model_radiomics.fit(X_train, Y_train)
        # best_model_feature_selection = best_model_feature_selection.fit(X_train_feature_selection, Y_train)

        trial_results = {'Trial #': [t], 'Training AUC': [training_auc], 'Validation AUC': [validation_auc],
                         'SF Testing AUC': [testing_auc]}
        df_trial = pd.DataFrame(trial_results)
        df_result = pd.concat([df_result, df_trial], axis=0, ignore_index=True)
        print("\r", f"Trial: {t + 1}/{num_trials}...Completed. Best score: {best_score_radiomics}.\n", end="")
    return df_result


# run
if __name__ == '__main__':
    random_seed(1, True)
    pd.set_option('display.max_rows', None)

    # Parameters
    num_trials = 10
    k = 5  # number of folds for cross-validation
    n_important_features = 50
    feature_selection_method = None  # naive, rfe    - mann whitney u? chi2? selectkbest? spearman rho? anova?
    include_location_2 = False  ################# how does this work? why adding SK OHE to Stanford and then concat with df_stanford?

    grid_parameters = {
        'n_estimators': [25, 50, 100],
        'random_state': [42, 100],
        'criterion': ["entropy"],
        'min_samples_leaf': [2, 4, 8],
        'max_depth': [1, 2, 4],
        'max_features': ["auto", None],
        'max_samples': [0.5, 0.75, 1]
    }

    df_sickkids = load_data(r'C:\Users\Justin\Documents\SickKids\pLGG\Data\Nomogram_study_LGG_data_Nov.27.xlsx',
                            sheet='SK')
    print(f'Rows: {df_sickkids.shape[0]}, Columns: {df_sickkids.shape[1]}')
    df_stanford = load_data(r'C:\Users\Justin\Documents\SickKids\pLGG\Data\Nomogram_study_LGG_data_Nov.27.xlsx',
                            sheet='Stanford')
    print(f'Rows: {df_stanford.shape[0]}, Columns: {df_stanford.shape[1]}')
    df_stanford_new = load_data(r'C:\Users\Justin\Documents\SickKids\pLGG\Data\Stanford_new_data_09_21.xlsx')
    print(f'Rows: {df_stanford_new.shape[0]}, Columns: {df_stanford_new.shape[1]}')

    print("Done loading data.\n")

    df_sickkids_processed, all_location_2_OHEs = preprocess_sickkids(df_sickkids, include_location_2)
    print(f'SickKids Data - Rows: {df_sickkids_processed.shape[0]}, Columns: {df_sickkids_processed.shape[1]}')
    print("SickKids data processed.\n")

    df_stanford_processed, df_stanford_new_processed, all_location_2_OHEs = preprocess_stanford(df_stanford,
                                                                                                df_stanford_new,
                                                                                                include_location_2,
                                                                                                all_location_2_OHEs)
    print(f'Stanford Data - Rows: {df_stanford_processed.shape[0]}, Columns: {df_stanford_processed.shape[1]}')
    print(
        f'Stanford_new Data - Rows: {df_stanford_new_processed.shape[0]}, Columns: {df_stanford_new_processed.shape[1]}')

    df_stanford_new_processed.columns = df_stanford_processed.columns
    df_stanford_combined_processed = pd.concat([df_stanford_processed, df_stanford_new_processed], ignore_index=True)
    print(
        f"Combined Stanford Data - Rows: {df_stanford_combined_processed.shape[0]}, Columns: {df_stanford_combined_processed.shape[1]}")
    print("Stanford data processed.\n")

    print(f"Total number of trial(s): {num_trials}, beginning experiment...")
    results = execute_experiment(num_trials=num_trials, k=k, grid_parameters=grid_parameters,
                                 df_SK=df_sickkids_processed,
                                 df_SF=df_stanford_combined_processed, OHEs=all_location_2_OHEs,
                                 location_2=include_location_2,
                                 feature_selection_method=feature_selection_method)
    print(f"{num_trials} trial(s) completed, experiment over.")

    curr_directory = os.path.dirname(os.path.realpath(__file__))
    sub_directory = "results"
    file_name = "RFC_results_" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".csv"
    filepath = os.path.join(curr_directory, sub_directory, file_name)
    results.to_csv(filepath, index=False)
