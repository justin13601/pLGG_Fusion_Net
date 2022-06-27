"""
Created on Aug 6

@author: Kareem

Task: Fusion vs other
Train both a CNN and radiomics model separately
Combine the two predictions to see if that improves the AUC
"""
import numpy as np
import pandas as pd
from os.path import dirname as up
import torch
import random
import socket
import nibabel as nib
import torch.nn as nn
import time
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from functools import partial
import torch.nn.functional as F
import pickle

class CustomImageDataset(Dataset):
    def __init__(self, data, patient_ids):
        self.data = data
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return self.data[self.patient_ids[idx]]["input"], self.data[self.patient_ids[idx]]["radiomics_input"],\
               self.data[self.patient_ids[idx]]["label"]



def get_inplanes():
    return [64, 128, 256, 512]


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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

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


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def load_pretrained_model(model, pretrain_path):
    pretrain = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])

    return model

def load_data_for_patient(patient_num):
    os.chdir(data_dir)
    os.chdir(os.path.join(data_dir, str(patient_num), "axflair"))
    files = os.listdir()
    image_file = None
    seg_file = None
    for f in files:
        if "_biasN4_bet.nii.gz" in f:
            image_file = f
        if "_REGISTERED_SEG.nii.gz" in f:
            seg_file = f
        else:
            continue
    image = nib.load(os.path.join(data_dir, str(patient_num), "axflair", image_file)).get_fdata()
    image = np.divide(image - np.amin(image), np.amax(image) - np.amin(image))
    mask = nib.load(os.path.join(data_dir, str(patient_num), "axflair", seg_file)).get_fdata()
    input = torch.tensor(np.multiply(image, mask)).float().unsqueeze(0)
    label = training_labels[patient_num]
    label = torch.tensor(label).float().unsqueeze(0)
    radiomics_input = torch.from_numpy(radiomic_features[patient_num].astype('float32'))
    result = {
        "input": input,
        "radiomics_input": radiomics_input,
        "label": label
    }
    return result

# Seeding
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    begin = time.time()
    epochs = 100
    num_trials = 20
    batch_size = 8
    lr=0.01
    half_dataset = False
    pretrained = True
    use_scheduler = False
    dropout_rate = 0.5 # default
    param_grid = {
            'n_estimators': [25, 50, 100],
            'random_state': [42, 100],
            'criterion': ["entropy"],
            'min_samples_leaf': [2, 4, 8],
            'max_depth': [1, 2, 4],
            'max_features': ["auto", None],
            'max_samples': [0.5, 0.75, 1]
        }

    ############################################Data Preprocessing######################################################
    # Pointing the excel file which contain the data labels
    if socket.gethostname()=="kareem-XPS-13-9310":
        root_dir = up(up(up(up(__file__))))
        data_path = os.path.join(root_dir, "Data", "Nomogram", "Nomogram_study_LGG_data_Nov.27.xlsx")
        data_SK_input = pd.read_excel(data_path, sheet_name="SK")
        data_dir = "/media/shared/Projects/SickKids_Brain_Preprocessing/Scans_non_rigid_fixed_origin_bs"
    else:
        data_SK_input = pd.read_excel("/media/Datasets/MedicalImages/BrainData//SickKids//Radiomics/Nomogram/Nomogram_study_LGG_data_Nov.27.xlsx", sheet_name="SK",engine="openpyxl")
        data_dir = "/media/Projects/SickKids_Brain_Preprocessing/Scans_non_rigid_fixed_origin_bs"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #######################################Preprocessing the radiomics spreadsheet######################################
    # Removing the last rows (which do not belong to examples)
    nanmask = np.isnan(data_SK_input["code"])
    data_SK = data_SK_input[~nanmask]
    data_SK = data_SK.reindex()

    # Remove patients that are to be excluded from the study
    excluded_patients = [9.0, 12.0, 23.0, 37.0, 58.0, 74.0, 78.0, 85.0, 121.0, 122.0, 130.0, 131.0, 138.0, 140.0, 150.0,
                         171.0, 176.0, 182.0, 204.0, 213.0, 221.0, 224.0, 234.0, 245.0, 246.0, 274.0, 306.0, 311.0,
                         312.0, 330.0, 334.0, 347.0, 349.0, 352.0, 354.0, 359.0, 364.0, 377.0,
                         235.0, 243.0, 255.0, 261.0, 264.0, 283.0, 288.0, 293.0,
                              299.0, 309.0, 325.0, 327.0, 333.0, 334.0, 356.0, 367.0,
                              376.0, 383.0, 387.0]
    data_SK = data_SK[~data_SK["code"].isin(excluded_patients)]
    data_SK = data_SK.reindex()

    # Remove data that we don't need for this analysis
    data_SK = data_SK.drop(columns = ['WT', 'NF1',
                                      'CDKN2A (0=balanced, 1=Del, 2=Undetermined)', 'FGFR 1', 'FGFR 2', 'FGFR 4',
                                      'Further gen info', 'Notes', 'Pathology Dx_Original', 'Pathology Coded',
                                      'Location_1', 'Location_2', 'Location_Original',  'Gender', 'Age Dx'])

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
    patient_codes = [int(x) for x in list(data_SK["code"].values)]
    training_labels = dict(zip(patient_codes,list(data_SK["label"].values)))
    data_SK = data_SK.drop(columns=["label"])

    # Organize the radiomic features into a dictionary with patient codes and corresponding patient features
    data_SK.set_index("code", inplace=True)
    radiomic_features = {}
    for index, row in data_SK.iterrows():
        radiomic_features[index] = row.values


    # Find the list of patients that we want to use: they haven't been excluded, and we have ax flair image
    patients_included = set(training_labels.keys())
    patients_with_desired_data = []
    for d in os.listdir(data_dir):
        try:
            patients_with_desired_data.append(int(d))
        except:
            continue
    patients_to_use = list(patients_included.intersection(patients_with_desired_data))
    print(len(patients_to_use))
    print(f"Start up time {time.time()- begin}")

    load_data_time = time.time()
    # Load the dataset into memory
    data = {}
    if socket.gethostname()=="kareem-XPS-13-9310":
        new_patients_to_use =[]
        for i in range(20):
            data[patients_to_use[i]]= load_data_for_patient(patients_to_use[i])
            new_patients_to_use.append(patients_to_use[i])
        patients_to_use = new_patients_to_use # Keep track of the patients we are acutally using here
    else:
        for i in range(len(patients_to_use)):
            data[patients_to_use[i]]= load_data_for_patient(patients_to_use[i])
    print(f"Time to load data into memory: {time.time() - load_data_time}")

    ###############################################Training#############################################################

    validation_aucs = []
    test_aucs = []
    best_epochs = []
    trial_time = []
    radiomics_auc = []
    combined_aucs = []

    # Keep track of predictions for both models, so we can use them after this to try and fit a better combined model
    predictions_and_labels = {}

    for t in range(num_trials):
        time_being_trial = time.time()
        # Set the seed for this iteration
        if t == 0:
            random_seed(17, True)
            next_seed = random.randint(0,1000)
        else:
            random_seed(next_seed, True)
            next_seed = random.randint(0, 1000)

        # Prepare the data loaders
        dataset = CustomImageDataset(data, patients_to_use)

        if half_dataset == False:
            # Use all data
            train_size = int(0.60 * len(dataset)) #120
            validation_size = int(0.2 * len(dataset)) #40
            test_size = len(dataset) - train_size - validation_size #40
            train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
            print(train_size)
            print(validation_size)
            print(test_size)
        else:
            # Use half data in train and val
            train_size = int(0.30 * len(dataset)) #60
            ignore_size = int(0.40 * len(dataset)) #80
            validation_size = int(0.1 * len(dataset)) #20
            test_size = len(dataset) - train_size - validation_size - ignore_size #40
            train_dataset, validation_dataset, test_dataset, ignore_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size, ignore_size])
            print(train_size)
            print(validation_size)
            print(test_size)
            print(ignore_size)


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Let's train the random forest on the radiomic features first
        # Use the dataloaders to create the train and test data for the random forest
        v = 0
        for inputs, radiomics_input, label in train_dataloader:
            if v ==0 :
                X_train = radiomics_input.numpy()
                Y_train = label.numpy()
            else:
                X_train = np.concatenate((X_train,radiomics_input))
                Y_train = np.concatenate((Y_train, label))
            v+=1
        for inputs, radiomics_input, label in validation_dataloader:
            X_train = np.concatenate((X_train, radiomics_input))
            Y_train = np.concatenate((Y_train, label))
        v = 0
        for inputs, radiomics_input, label in test_dataloader:
            if v ==0 :
                X_test = radiomics_input.numpy()
                Y_test= label.numpy()
            else:
                X_test = np.concatenate((X_test,radiomics_input))
                Y_test = np.concatenate((Y_test, label))
            v+=1

        # Train the model
        # Fit random forest on all radiomics features
        se = np.random.randint(10000)  # New seed for splitting data each trial
        kf = KFold(n_splits=5, shuffle=True, random_state=se)
        rfc_cv = RandomForestClassifier(n_jobs=-1)
        CV_rfc = GridSearchCV(estimator=rfc_cv, param_grid=param_grid, cv=kf, verbose = 0, scoring="neg_log_loss")
        temp = CV_rfc.fit(X_train, Y_train.ravel())
        best_model_original = temp.best_estimator_
        best_score_original = CV_rfc.best_score_
        Y_test_estimated = best_model_original.predict_proba(X_test)[:,1]
        result_radiomics = round(roc_auc_score(Y_test, Y_test_estimated), 3)
        radiomics_auc.append(result_radiomics)
        print(f"Radiomics AUC: {result_radiomics}")

        # Define neural network stuff
        # Need to change directory back to where the saved model is located
        if socket.gethostname() == "kareem-XPS-13-9310":
            continue
        else:
            os.chdir("/media/Kareem")
        net = generate_model(model_depth=18, n_classes=1039)
        if pretrained:
            print('loading pretrained model: r3d18_KM_200ep.pth')
            net = load_pretrained_model(net, "r3d18_KM_200ep.pth")
        else:
            print('not using a pretrained model')
        # Adjust the number of inputs and outputs
        net.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        net.fc = net.fc = nn.Linear(512, 1)
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

        # Variables for tracking results-flair
        lowest_val_loss = 1000.0 # Lowest loss on validation set
        lowest_val_loss_epoch = 0 # Epoch on which lowest loss on validation set was acheived
        best_val_auc = None # The AUC on the validation set for the model that had the best validation loss
        best_test_auc = None # The AUC on the validation set for the model that had the best validation loss
        best_combined_auc = None # The AUC on the test set for the model that had the best validation loss

        # Keep track of data we need to train the combined random forest model
        best_test_preds_nn = None  # The predictions on the test set for the model that had the best validation loss
        best_test_preds_rad = None  # The predictions from the radiomics corresponding to the above
        best_test_preds_labels = None  # The labels corresponding to the above
        best_dev_preds_nn = None  # The predictions on the dev set for the model that had the best validation loss
        best_dev_preds_rad = None  # The predictions from the radiomics corresponding to the above
        best_dev_preds_labels = None  # The labels corresponding to the above

        for epoch in range(epochs):
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            net.train()
            train_loss = 0
            train_batches = 0
            training_true = []
            training_estimated = []
            train_radiomics_estimated =[]
            for inputs, radiomics_input, label in train_dataloader:
                # Put data on GPU
                inputs = inputs.to(device)
                label = label.to(device)

                # Add noise to images
                noise = torch.randn_like(inputs, device=device)*0.1
                inputs = inputs+noise

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()

                for i in range(len(label.tolist())):
                    training_true.append(label.tolist()[i][0])
                    training_estimated.append(output.tolist()[i][0])

                # Keep track of loss through the entire epoch
                train_loss += loss.item()*batch_size
                train_batches +=1

                # Get predictions for the radiomics model
                train_radiomics_estimated.extend(best_model_original.predict_proba(radiomics_input)[:, 1])

            # Calculate average loss over epoch
            train_loss = train_loss/(train_batches*batch_size)

            # Get results-flair on the validation and test sets
            net.eval()
            with torch.set_grad_enabled(False):
                # Validation
                val_loss = 0
                batches = 0
                validation_true = []
                validation_estimated = []
                validation_radiomics_estimated = []
                for inputs, radiomics_input, label in validation_dataloader:
                    # Transfer to GPU
                    inputs, label = inputs.to(device), label.to(device)
                    output = net(inputs)
                    for i in range(len(label.tolist())):
                        validation_true.append(label.tolist()[i][0])
                        validation_estimated.append(output.tolist()[i][0])
                    validation_radiomics_estimated.extend(best_model_original.predict_proba(radiomics_input)[:, 1])
                    val_loss += criterion(output, label).item()*batch_size
                    batches+=1
                val_loss = val_loss / (batches * batch_size)

                # Test
                test_true = []
                test_estimated = []
                test_radiomics_estimated = []
                for inputs, radiomics_input, label in test_dataloader:
                    # Transfer to GPU
                    inputs, label = inputs.to(device), label.to(device)
                    output = net(inputs)
                    for i in range(len(label.tolist())):
                        test_true.append(label.tolist()[i][0])
                        test_estimated.append(output.tolist()[i][0])
                    test_radiomics_estimated.extend(best_model_original.predict_proba(radiomics_input)[:,1])

            # Calculate the AUC for the different models
            val_auc = roc_auc_score(validation_true, validation_estimated)
            test_auc = roc_auc_score(test_true, test_estimated)
            train_auc = roc_auc_score(training_true, training_estimated)
            combined_auc = roc_auc_score(test_true,(np.array(test_estimated)+np.array(test_radiomics_estimated))/2)


            print(f"trial: {t}, epoch: {epoch}, "
                  f"training loss {round(train_loss,3)} , "
                  f"validation loss: {round(val_loss,3)}, "
                  f"training AUC: {round(train_auc,3)}, "
                  f"validation AUC: {round(val_auc,3)}, "
                  f"test AUC: {round(test_auc,3)}, "
                  f"combined AUC: {round(combined_auc,3)}")


            if val_loss< lowest_val_loss:
                lowest_val_loss = val_loss
                lowest_val_loss_epoch = epoch
                best_test_auc = test_auc
                best_val_auc = val_auc
                best_combined_auc = combined_auc
                best_test_preds_nn = test_estimated
                best_test_preds_rad = test_radiomics_estimated
                best_test_preds_labels = test_true
                best_dev_preds_nn = training_estimated + validation_estimated
                best_dev_preds_rad = train_radiomics_estimated + validation_radiomics_estimated
                best_dev_preds_labels = training_true + validation_true

            if epoch == epochs - 1:
                validation_aucs.append(round(best_val_auc,3))
                test_aucs.append(round(best_test_auc,3))
                combined_aucs.append(round(best_combined_auc,3))
                best_epochs.append(round(lowest_val_loss_epoch,3))


            if epoch-lowest_val_loss_epoch>=10:
                validation_aucs.append(round(best_val_auc,3))
                test_aucs.append(round(best_test_auc,3))
                combined_aucs.append(round(best_combined_auc, 3))
                best_epochs.append(round(lowest_val_loss_epoch,3))
                break

        print(f"Summary for trial {t}")
        print(f"Radiomics AUC: {round(roc_auc_score(best_test_preds_labels, best_test_preds_rad), 3)}")
        print(f"Neural Network AUC: {round(roc_auc_score(best_test_preds_labels, best_test_preds_nn), 3)}")
        print(f"Naive Combine AUC: {round(roc_auc_score(best_test_preds_labels, (np.array(best_test_preds_nn)+np.array(best_test_preds_rad))/2), 3)}")
        print(f"Time for this trial: {round(time.time() - time_being_trial,3)}")
        trial_time.append(round(time.time() - time_being_trial,3))

        # Keep track of all predictions
        predictions_and_labels[t] = {"neural_net_test_preds":best_test_preds_nn,
                                     "radiomics_test_preds":best_test_preds_rad,
                                     "test_labels":best_test_preds_labels,
                                     "neural_net_dev_preds": best_dev_preds_nn,
                                     "radiomics_dev_preds": best_dev_preds_rad,
                                     "dev_labels": best_dev_preds_labels,
                                     "AUC_nn":round(roc_auc_score(best_test_preds_labels, best_test_preds_nn), 3),
                                     "AUC_rad":round(roc_auc_score(best_test_preds_labels, best_test_preds_rad), 3),
                                     "AUC_naive_combine":round(roc_auc_score(best_test_preds_labels, (np.array(best_test_preds_nn)+np.array(best_test_preds_rad))/2), 3)}

    # Save dictionary of predictions
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open("/home/kkudus/resnet_and_radiomics_data" + "_" + timestamp + ".pk", 'wb') as handle:
        pickle.dump(predictions_and_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Radiomics AUCs: {radiomics_auc}, mean: {np.mean(radiomics_auc)}, median: {np.median(radiomics_auc)}")
    print(f"Validation AUCs: {validation_aucs}, mean: {np.mean(validation_aucs)}, median: {np.median(validation_aucs)}")
    print(f"Test AUCs: {test_aucs}, mean: {np.mean(test_aucs)}, median: {np.median(test_aucs)}")
    print(f"Combined AUCs: {combined_aucs}, mean: {np.mean(combined_aucs)}, median: {np.median(combined_aucs)}")
    print(f"Time for the trials: {trial_time}, mean: {np.mean(trial_time)}, median: {np.mean(trial_time)}")
    print(f"Epoch with best model: {best_epochs}")
    results = pd.DataFrame(
        columns=["Validation AUCs", "Test AUCs", "Radiomics AUCs", "Combined AUCs" ,
                 "Times", "Best Epochs"])
    for i in range(len(validation_aucs)):
        result = []
        result.append(validation_aucs[i])
        result.append(test_aucs[i])
        result.append(radiomics_auc[i])
        result.append(combined_aucs[i])
        result.append(trial_time[i])
        result.append(best_epochs[i])
        results.loc[len(results)] = result


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results)

    name = "/home/kkudus/resnet_and_radiomics_results_flair_pretrained" + str(pretrained) + "_dropout" + str(dropout_rate) +\
           "_halfdataset" + str(half_dataset) + "_lr" + str(lr) + "_scheduler" + str(use_scheduler) +\
           "_" + timestamp + ".csv"
    print(name)
    results.to_csv(name)