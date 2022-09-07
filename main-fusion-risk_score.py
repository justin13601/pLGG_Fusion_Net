import os
import sys
import csv
import time
import math
import glob
import socket
import logging
import random
import numpy as np
import pandas as pd
from google.cloud import bigquery

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from functools import partial


# %%
def load_excel_data(path, sheet=0):
    filename = os.path.basename(path).strip()
    if isinstance(sheet, str):
        logging.info(f'Loading {filename}, Sheet: {sheet}...')
    else:
        logging.info('Loading ' + filename + '...')
    df_data = pd.read_excel(path, sheet)
    logging.info("Done loading.")
    return df_data


def load_image_data(path, patients, limit=False):
    data_images = {}
    for root, dirs, files in os.walk(path):
        dirs.sort(key=int)
        dirs = list(map(int, dirs))
        dirs = [patient for patient in dirs if patient in patients]
        if limit:
            dirs = dirs[:limit]
        for d in dirs:
            logging.info(f"Loading Patient {d}...")
            np_filenames = glob.glob(f"{os.path.join(root, f'{d}')}/*/*.npy")
            data_images[d] = [np.load(np_filenames[0]), np.load(np_filenames[1])]
        break
    return data_images, dirs


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


# %%
###############################################################################
# Plot Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err)  # number of epochs
    plt.plot(range(1, n + 1), train_err, label="Train")
    plt.plot(range(1, n + 1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1, n + 1), train_loss, label="Train")
    plt.plot(range(1, n + 1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


def plot_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()
    return


###############################################################################
# %%
# Model Classes
class CNNDataset(Dataset):
    def __init__(self, data, patient_ids):
        self.data = data
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return self.data[self.patient_ids[idx]]["input"], self.data[self.patient_ids[idx]]["label"], \
               self.data[self.patient_ids[idx]]["features"]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.name = "BasicBlock"
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.name = "Bottleneck"
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# %%
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 model_depth,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.name = f"ResNet_pLGG_Classifer_depth{model_depth}"

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x


# %%

def generate_model(model_depth, inplanes, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    model = None
    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], inplanes, model_depth, widen_factor=widen_factor, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], inplanes, model_depth, widen_factor=widen_factor, **kwargs)

    return model


def get_model_name(trial, name, batch_size, learning_rate, dropout_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "trial_{0}_model_{1}_bs{2}_lr{3}_dr{4}_epoch{5}".format(trial,
                                                                   name,
                                                                   batch_size,
                                                                   learning_rate,
                                                                   dropout_rate,
                                                                   epoch)
    return path


# %%

########################################################################
# Other functions
def create_label(mutation, fusion):
    if mutation == 1:
        return 1
    elif fusion == 1:
        return 0
    else:
        return None


def process_excel(df_data, exclusions, custom_features=None):
    nanmask = np.isnan(df_data["code"])
    data_data_new = df_data[~nanmask]
    data_data_new = data_data_new.reindex()

    # Remove exluded patients
    data_data_new = data_data_new[~data_data_new["code"].isin(exclusions)]
    data_data_new = data_data_new.reindex()

    # Remove data that we don't need for this analysis
    data_data_new = data_data_new.drop(columns=['WT', 'NF1',
                                                'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1', 'FGFR 2',
                                                'FGFR 4',
                                                'Further gen info', 'Notes', 'Pathology Dx_Original', 'Pathology Coded',
                                                'Location_1', 'Location_2', 'Location_Original', 'Gender', 'Age Dx'])

    data_data_new['label'] = data_data_new.apply(lambda x: create_label(x['BRAF V600E final'], x['BRAF fusion final']),
                                                 axis=1)
    data_data_new = data_data_new.drop(columns=["BRAF V600E final", "BRAF fusion final"])

    # Drop rows where the outcome is not mutation or fusion
    nanmask = np.isnan(data_data_new["label"])
    data_data_new = data_data_new[~nanmask]
    data_data_new = data_data_new.reindex()
    patient_codes = [int(x) for x in list(data_data_new["code"].values)]

    training_labels = dict(zip(patient_codes, list(data_data_new["label"].values)))
    data_data_new = data_data_new.drop(columns=["label"])

    # Organize the radiomic features into a dictionary with patient codes and corresponding patient features
    if custom_features is not None:
        radiomic_features = {}
        for index, row in custom_features.iterrows():
            radiomic_features[custom_features['id'][index]] = row.values[1:]
    else:
        data_data_new.set_index("code", inplace=True)
        radiomic_features = {}
        for index, row in data_data_new.iterrows():
            radiomic_features[index] = row.values
    return radiomic_features, training_labels


# %%

###############################################################################
# Model Training
def evaluate(net, loader, criterion=nn.BCEWithLogitsLoss()):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    net.eval()
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    true = []
    estimated = []
    for i, data in enumerate(loader, 0):
        inputs, labels, features = data
        # print(labels.float())
        # print(data)
        # print(i)
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            net = net.cuda()
        outputs = net(inputs)
        # print(outputs.float())
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

        for i in range(len(labels.tolist())):
            true.append(labels.tolist()[i][0])
            estimated.append(outputs.tolist()[i][0])
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss, true, estimated


# %%
def remove_correlated_features(X=None, threshold=0.95):
    if X is not None:
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        X = X.drop(to_drop, axis='columns')
    return X


def variance_threshold(X):
    selector = VarianceThreshold(threshold=0.05)
    ids = X['id']
    X = selector.fit_transform(X.loc[:, X.columns != 'id'])
    X = pd.DataFrame(X)
    X['id'] = ids
    return X


# %%

# %%
def train_net(dev_dataset, test_dataset, trial, net, optimizer, criterion, batch_size=64,
              learning_rate=0.01, num_epochs=30, checkpoint=False,
              save_folder=os.getcwd(), radiomics_grid_parameters=dict, fusion_grid_parameters=dict):
    training_start_time = time.time()

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # fit radiomics model on dev data
    rfc = RandomForestClassifier(n_jobs=-1)  # all CPUs used
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=radiomics_grid_parameters, verbose=0, scoring="roc_auc",
                          n_jobs=-1, return_train_score=True, refit=True)

    dev_patient_data = [patient_data for patient_data in dev_dataset]
    Y_dev = np.concatenate([patient_data[1].numpy() for patient_data in dev_patient_data])

    test_patient_data = [patient_data for patient_data in test_dataset]
    X_test = pd.DataFrame([patient_data[2].numpy().squeeze() for patient_data in test_patient_data])
    Y_test = np.concatenate([patient_data[1].numpy() for patient_data in test_patient_data])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    all_dev_predictions_from_radiomics = {}
    all_test_predictions_from_radiomics = {}
    all_dev_predictions_from_resnet = {}
    all_test_predictions_from_resnet = {}

    all_dev_true_split_by_fold = {}
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dev_dataset)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=val_sampler)

        all_train_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), sampler=train_sampler)
        all_val_loader = DataLoader(dev_dataset, batch_size=len(dev_dataset), sampler=val_sampler)

        # RADIOMICS
        train_patient_data = [patient_data for patient_data in all_train_loader]
        X_train = pd.DataFrame([d[2].numpy().squeeze() for d in train_patient_data][0])
        Y_train = np.concatenate([d[1].numpy() for d in train_patient_data][0])
        fit_model = rfc_cv.fit(X_train, Y_train)
        best_model_radiomics = fit_model.best_estimator_

        # get val predictions for best radiomics model
        val_patient_data = [patient_data for patient_data in all_val_loader]
        X_val = pd.DataFrame([d[2].numpy().squeeze() for d in val_patient_data][0])
        Y_val = np.concatenate([d[1].numpy() for d in val_patient_data][0])
        all_dev_predictions_from_radiomics[fold] = best_model_radiomics.predict_proba(X_val)[:, 1]

        all_test_predictions_from_radiomics[fold] = best_model_radiomics.predict_proba(X_test)[:, 1]

        # CNN
        train_history = {'train_auc': np.zeros(num_epochs), 'train_loss': np.zeros(num_epochs),
                         'val_auc': np.zeros(num_epochs), 'val_loss': np.zeros(num_epochs),
                         'test_auc': np.zeros(num_epochs), 'test_loss': np.zeros(num_epochs)}

        dev_predictions_from_resnet_epochs = {}
        test_predictions_from_resnet_epochs = {}

        model_results_name = get_model_name(trial=trial, name=net.name, batch_size=batch_size,
                                            learning_rate=learning_rate, dropout_rate=dropout_rate, epoch=num_epochs)
        for epoch in range(num_epochs):
            if len(train_history['val_loss']) > 3:
                if train_history['val_loss'][-1] > train_history['val_loss'][-2] > train_history['val_loss'][-3]:
                    break

            epoch_start_time = time.time()

            # Training
            # train_err = 0.0
            net.train()
            train_loss = 0.0
            total_epoch = 0
            training_true = []
            training_estimated = []
            n = 0
            for inputs, labels, features in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Add noise to images
                noise = torch.randn_like(inputs, device=device) * 0.1
                inputs = inputs + noise

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                if use_scheduler:
                    scheduler.step()

                # corr = (outputs > 0.0).squeeze().long() != labels
                # train_err += int(corr.sum())
                # Keep track of loss through the entire epoch
                train_loss += loss.item()
                total_epoch += len(labels)
                n = n + 1

                for i in range(len(labels.tolist())):
                    training_true.append(labels.tolist()[i][0])
                    training_estimated.append(outputs.tolist()[i][0])

            # Calculate average over epoch
            # total_train_err[epoch] = float(train_err) / total_epoch
            train_history['train_loss'][epoch] = float(train_loss) / n

            # Validation
            eval_err_val, eval_loss_val, validation_true, validation_estimated = evaluate(net, val_loader)
            train_history['val_loss'][epoch] = eval_loss_val

            eval_err_test, eval_loss_test, test_true, test_estimated = evaluate(net, test_loader)
            train_history['test_loss'][epoch] = eval_loss_test

            dev_predictions_from_resnet_epochs[epoch] = validation_estimated
            test_predictions_from_resnet_epochs[epoch] = test_estimated

            # Calculate the AUC for the different models
            train_auc = roc_auc_score(training_true, training_estimated)
            val_auc = roc_auc_score(validation_true, validation_estimated)
            test_auc = roc_auc_score(test_true, test_estimated)

            train_history['train_auc'][epoch] = train_auc
            train_history['val_auc'][epoch] = val_auc
            train_history['test_auc'][epoch] = test_auc

            logging.info(
                "Epoch {}: Train loss: {}, Train AUC: {} | Val loss: {}, Val AUC: {} | Test loss: {}, Test AUC: {}".format(
                    epoch + 1,
                    train_history['train_loss'][epoch],
                    train_history['train_auc'][epoch],
                    train_history['val_loss'][epoch],
                    train_history['val_auc'][epoch],
                    train_history['test_loss'][epoch],
                    train_history['test_auc'][epoch]))

            if checkpoint:
                torch.save(net.state_dict(), model_results_name)

        all_epochs_val_loss = train_history['val_loss']
        best_epoch = np.where(all_epochs_val_loss == min(all_epochs_val_loss))[0][0]
        print(best_epoch)

        all_dev_predictions_from_resnet[fold] = dev_predictions_from_resnet_epochs[best_epoch]
        all_test_predictions_from_resnet[fold] = test_predictions_from_resnet_epochs[best_epoch]

        all_dev_true_split_by_fold[fold] = Y_val

    rfc_fusion = RandomForestClassifier(n_jobs=-1)  # all CPUs used
    rfc_cv_fusion = GridSearchCV(estimator=rfc_fusion, param_grid=fusion_grid_parameters, cv=kfold, verbose=0,
                                 scoring="roc_auc", n_jobs=-1, return_train_score=True, refit=True)

    val_aucs_by_fold_radiomics = []
    for fold in all_dev_predictions_from_radiomics.keys():
        val_aucs_by_fold_radiomics.append((fold, roc_auc_score(all_dev_true_split_by_fold[fold],
                                                               all_dev_predictions_from_radiomics[fold])))
    best_fold_radiomics = min(val_aucs_by_fold_radiomics, key=lambda x: x[1])[0]
    test_estimated_radiomics_best_fold = all_test_predictions_from_radiomics[best_fold_radiomics]

    val_aucs_by_fold_resnet = []
    for fold in all_dev_predictions_from_resnet.keys():
        val_aucs_by_fold_resnet.append((fold, roc_auc_score(all_dev_true_split_by_fold[fold],
                                                            all_dev_predictions_from_resnet[fold])))
    best_fold_resnet = min(val_aucs_by_fold_resnet, key=lambda x: x[1])[0]
    test_estimated_resnet_best_fold = all_test_predictions_from_resnet[best_fold_resnet]

    dev_predictions_radiomics = []
    for fold in all_dev_predictions_from_radiomics.keys():
        dev_predictions_radiomics = dev_predictions_radiomics + all_dev_predictions_from_radiomics[fold]

    dev_predictions_resnet = []
    for fold in all_dev_predictions_from_resnet.keys():
        dev_predictions_resnet = dev_predictions_resnet + all_dev_predictions_from_resnet[fold]

    X_fusion_dev = pd.DataFrame(
        list(zip(dev_predictions_radiomics, dev_predictions_resnet)),
        columns=['radiomics', 'resnet'])
    X_fusion_test = pd.DataFrame(
        list(zip(test_estimated_radiomics_best_fold, test_estimated_resnet_best_fold)),
        columns=['radiomics', 'resnet'])

    fit_model_fusion = rfc_cv_fusion.fit(X_fusion_dev, Y_dev)
    best_model_radiomics_fusion = fit_model_fusion.best_estimator_

    # get predictions for fusion RFC model
    X_fusion_test_predicted = best_model_radiomics_fusion.predict_proba(X_fusion_test)

    dev_auc = fit_model_fusion.cv_results_['mean_train_score'][fit_model_fusion.best_index_]
    test_auc = roc_auc_score(Y_test, X_fusion_test_predicted[:, 1])

    np.savetxt(os.path.join(save_folder, "{}_dev_fusion_auc.csv".format(model_results_name)), dev_auc)
    np.savetxt(os.path.join(save_folder, "{}_test_fusion_auc.csv".format(model_results_name)), test_auc)

    logging.info('Finished training.')
    logging.info(f'Time elapsed: {round(time.time() - training_start_time, 3)} seconds.')

    return dev_auc, test_auc


########################################################################
# Currently only SK images are used, so "test_dataloader" represents
# internal testing data. Once Stanford images are processed, they will
# also be a "testing set" but rather "external testing data".
########################################################################
# run
if __name__ == '__main__':
    start_up_time = time.time()

    save_folder = os.path.join(os.getcwd(), f"CNN_results_{time.strftime('%Y_%m_%d-%H_%M_%S')}")
    os.mkdir(save_folder)
    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(save_folder, 'output_log.log'))
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)

    random_seed(random.randint(1, 100), True)
    pd.set_option('display.max_rows', None)

    # use numpy files instead of .nii
    # no need to normalize images between [0,1] as input images are already preprocessed
    # https://github.com/kenshohara/3D-ResNets-PyTorch

    radiomics_directory = r'C:\Users\Justin\Documents\Data'
    image_directory = r'K:\Projects\SickKids_Brain_Preprocessing\preprocessed_all_seq_kk_july_2022'

    # Parameters
    load_model = False
    use_scheduler = False
    limit = 20

    num_trials = 2
    num_epochs = 2
    batch_size = 8
    learning_rate = 0.01
    dropout_rate = 0.5  # default
    widen_factor = 0.5  # speeds up training without affect performance too much
    inplanes = [64, 128, 256, 512]
    k = 5

    fusion_grid_parameters = {
        'n_estimators': [5, 10, 25],
        'random_state': [42, 100],
        'criterion': ["entropy"],
        'min_samples_leaf': [1, 2, 4],
        'max_depth': [1, 2],
        'max_features': [None],
        'max_samples': [0.5, 0.75, 1]
    }

    radiomics_grid_parameters = {
        'n_estimators': [25, 50, 100],
        'random_state': [42, 100],
        'criterion': ["entropy"],
        'min_samples_leaf': [2, 4, 8],
        'max_depth': [1, 2, 4],
        'max_features': [None],
        'max_samples': [0.5, 0.75, 1]
    }

    excluded_patients = [2, 3, 4, 6, 7, 9, 11, 12, 13, 16, 21, 23, 25, 28, 29, 30, 33, 35, 36, 37, 38, 44, 45, 49, 50,
                         52, 53, 54, 55, 58, 59, 61, 63, 66, 73, 74, 75, 77, 78, 80, 84, 85, 86, 92, 95, 96, 98, 100,
                         102, 103, 105, 107, 108, 110, 113, 117, 121, 122, 123, 125, 128, 130, 131, 132, 136, 137, 138,
                         139, 140, 141, 142, 143, 147, 148, 150, 152, 156, 158, 159, 165, 166, 171, 173, 174, 176, 182,
                         183, 184, 187, 190, 191, 192, 194, 196, 199, 203, 204, 209, 210, 213, 221, 222, 224, 226, 227,
                         228, 232, 233, 234, 235, 237, 240, 242, 243, 245, 246, 250, 254, 255, 256, 258, 259, 260, 261,
                         263, 264, 266, 270, 272, 274, 277, 278, 283, 284, 285, 288, 293, 298, 299, 303, 306, 309, 311,
                         312, 317, 318, 321, 322, 324, 325, 327, 328, 330, 332, 333, 334, 336, 337, 341, 343, 347,
                         349, 350, 351, 352, 354, 356, 359, 364, 367, 370, 371, 374, 376, 377, 378, 380, 383, 386, 387,
                         388, 392, 396, 243, 255, 261, 264, 288, 299, 309, 327, 351, 387]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load data
    df_sickkids = load_excel_data(os.path.join(radiomics_directory, 'Nomogram_study_LGG_data_Nov.27.xlsx'), sheet='SK')
    df_features = pd.read_csv(r'C:\Users\Justin\Documents\Data\radiomics_features_normalized_08-15-22_filtered_851.csv')

    df_features = remove_correlated_features(df_features)
    df_features = variance_threshold(df_features)

    sickkids_radiomics_features, sickkids_labels = process_excel(df_data=df_sickkids, exclusions=excluded_patients,
                                                                 custom_features=df_features)

    # Prepare CNN data
    radiomics_patients_list = set(sickkids_labels.keys())
    patients_with_FLAIR = []
    for each_patient in os.listdir(image_directory):
        try:
            patients_with_FLAIR.append(int(each_patient))
        except:
            logging.info(f'Patient {each_patient} FLAIR not found.')
    patients_with_FLAIR.sort(key=int)
    patients_list = list(radiomics_patients_list.intersection(patients_with_FLAIR))
    logging.info(f"Total number of patients: {len(patients_list)}.")
    logging.info(f"Start-up time: {round(time.time() - start_up_time, 3)} seconds.\n")

    load_image_time = time.time()
    images, patients_used = load_image_data(image_directory, patients=patients_list, limit=limit)
    data_pre_norm = {}
    for each_patient in patients_used:
        image = images[each_patient][0]
        seg = images[each_patient][1]
        data_pre_norm[each_patient] = (image, seg)
    data_post_norm = {}
    for each_patient in data_pre_norm.keys():
        image_pre_norm = data_pre_norm[each_patient][0]
        seg = data_pre_norm[each_patient][1]
        image_post_norm = (image_pre_norm - np.min(image_pre_norm)) / (np.max(image_pre_norm) - np.min(image_pre_norm))
        data_post_norm[each_patient] = (image_post_norm, seg)
    data = {}
    for each_patient in patients_used:
        input = torch.tensor(
            np.multiply(data_post_norm[each_patient][0], data_post_norm[each_patient][1])).float().unsqueeze(0)
        label = sickkids_labels[each_patient]
        label = torch.tensor(label).float().unsqueeze(0)
        features = torch.tensor(sickkids_radiomics_features[each_patient].tolist()).float().unsqueeze(0)
        patient = {
            "input": input,
            "label": label,
            "features": features
        }
        data[each_patient] = patient

    logging.info("Done loading images.")
    logging.info(f"Number of patients included: {len(patients_used)}.")
    logging.info(f"Image loading time: {round(time.time() - load_image_time, 3)} seconds.\n")

    # if load_model:
    #     try:
    #         net = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)
    #         model_path = get_model_name(trial=1, name=net.name, batch_size=batch_size, learning_rate=learning_rate,
    #                                     dropout_rate=dropout_rate, epoch=num_epochs)
    #         state = torch.load(model_path)
    #         net.load_state_dict(state)
    #     except FileNotFoundError:
    #         logging.info('Model not found.')
    #     else:
    #         logging.info("Insert code...")
    #     sys.exit()

    training_aucs = []
    validation_aucs = []
    test_aucs = []
    best_epochs = []
    trial_times = []

    df_result = pd.DataFrame(columns=['Development AUC', 'Testing AUC'])
    for t in range(num_trials):
        logging.info(f"Beginning trial {t + 1} of {num_trials}...")
        begin_trial_time = time.time()

        # Set the seed for this iteration
        if t == 0:
            random_seed(1, True)
            next_seed = random.randint(0, 1000)
        else:
            random_seed(next_seed, True)
            next_seed = random.randint(0, 1000)

        dataset = CNNDataset(data, patients_used)
        development_size = int(0.75 * len(dataset))
        test_size = len(dataset) - development_size
        development_dataset, test_dataset = torch.utils.data.random_split(dataset, [development_size, test_size])
        logging.info(f"Datasplit -> Development: {development_size}, Testing: {test_size}.")

        net = generate_model(model_depth=18, inplanes=inplanes, n_classes=1039)

        net.conv1 = nn.Conv3d(1, int(64 * widen_factor), kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3),
                              bias=False)
        net.fc = net.fc = nn.Linear(math.floor(512 * widen_factor), 1)

        net.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

        results = train_net(development_dataset, test_dataset,
                            trial=t + 1,
                            net=net,
                            optimizer=optimizer,
                            criterion=criterion,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_epochs=num_epochs, save_folder=save_folder,
                            radiomics_grid_parameters=radiomics_grid_parameters,
                            fusion_grid_parameters=fusion_grid_parameters)

        dev_fusion_auc, test_fusion_auc = results

        df_trial = pd.DataFrame(list(zip([dev_fusion_auc], [test_fusion_auc])),
                                columns=['Development', 'Testing'])
        df_result = pd.concat([df_result, df_trial], axis=0, ignore_index=True)

        logging.info("\r",
                     f"Trial: {t + 1}/{num_trials}...Completed. Fusion AUC on development set: {dev_fusion_auc}, fusion AUC on test set: {test_fusion_auc}.\n")

        trial_duration = time.time() - begin_trial_time
        trial_times.append(round(trial_duration, 3))
        logging.info(f"Trial {t + 1} ended. Duration: {round(trial_duration, 3)} seconds.\n")
    logging.info(f'Experiment done. Time elapsed: {round(time.time() - start_up_time, 3)} seconds.')

    curr_directory = os.path.dirname(os.path.realpath(__file__))
    sub_directory = "RFC_fusion_results"
    file_name = "RFC_fusion_results_" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".csv"
    filepath = os.path.join(curr_directory, sub_directory, file_name)
    df_result.to_csv(filepath, index=False)

logging.info('---------------------')
