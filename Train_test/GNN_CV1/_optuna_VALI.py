import os
import json
import time
import datetime
import random
import copy
import math
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(currentdir, "../../Model"))
sys.path.insert(1, currentdir + "/../..")
_func = __import__("_func")

# Helper function to get data
get_data = _func.get_data

# Check for GPU availability
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(trial_setup_dict):
    print("Model ...")
    GNN_model = __import__(trial_setup_dict['modeltype_pyfilename'])
    model = GNN_model.MODEL(trial_setup_dict['model'])
    return model

def create_optimizer(trial_setup_dict, model):
    print("Optimizer ...")
    optimizer_chosen = trial_setup_dict["optimizer_chosen"]
    kwargs = trial_setup_dict[optimizer_chosen]
    optimizer = getattr(optim, optimizer_chosen)(model.parameters(), **kwargs)
    return optimizer

def train_or_eval(model, optimizer, dataloader, loss_fn, mode="eval", training_seed=None):
    if mode == "train" and training_seed is None:
        print("ERROR, require to set a training seed")
        exit()

    epoch_loss = 0
    total = 0

    if mode == "train":
        model.train()
        torch.manual_seed(training_seed)
    else:
        model.eval()

    with torch.set_grad_enabled(mode == "train"):
        for features, labels in dataloader:
            features, labels = features.to(_device), labels.to(_device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            if mode == "train":
                loss.backward()
                optimizer.step()

            total += labels.size(0)
            epoch_loss += loss.item() * labels.size(0)

    return epoch_loss / total

def evaluate(model, test_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for features, labels in test_loader:
            features, labels = features.to(_device), labels.to(_device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)

            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        r_square = calculate_r_square(all_labels, all_preds)
        return total_loss / total_samples, r_square

def calculate_r_square(y_true, y_pred):
    linreg = LinearRegression().fit(np.array(y_pred).reshape(-1, 1), np.array(y_true))
    return linreg.score(np.array(y_pred).reshape(-1, 1), np.array(y_true))

def objective(trial, optuna_setup_dict, mode="trial"):
    print("Objective ...")

    if mode == "trial":
        for k, v in optuna_setup_dict.items():
            if isinstance(v, dict):
                v_type = list(v.keys())[0]
                if v_type == 'categorical':
                    trial.suggest_categorical(k, v[v_type])
                elif v_type == 'discrete_uniform':
                    trial.suggest_discrete_uniform(k, v[v_type][0], v[v_type][1], v[v_type][2])
                elif v_type in ['float', 'int']:
                    v_low = v[v_type][0]
                    v_high = v[v_type][1]
                    v_step = v[v_type][2].get('step') if len(v[v_type]) > 2 else None
                    v_log = v[v_type][2].get('log') if len(v[v_type]) > 2 else False
                    if v_type == 'float':
                        trial.suggest_float(k, v_low, v_high, step=v_step, log=v_log)
                    else:
                        trial.suggest_int(k, v_low, v_high, step=v_step, log=v_log)
                elif v_type in ['loguniform', 'uniform']:
                    if v_type == 'loguniform':
                        trial.suggest_loguniform(k, v[v_type][0], v[v_type][1])
                    else:
                        trial.suggest_uniform(k, v[v_type][0], v[v_type][1])
                else:
                    print("ERROR v_type {}", k, v)
                print("setting {} to param {}".format(v, k))
            else:
                trial.set_user_attr(k, v)

        def merge(d1, d2):
            for k in d2:
                if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    merge(d1[k], d2[k])
                else:
                    d1[k] = d2[k]

        def copy_merge(d1, d2):
            _d1 = copy.deepcopy(d1)
            _d2 = copy.deepcopy(d2)
            merge(_d1, _d2)
            return _d1

        trial_setup_dict = {}
        for _dict in [trial.user_attrs, trial.params]:
            for k, v in _dict.items():
                if "|" in k:
                    _kname_list = k.split("|")
                    _kname_list.reverse()
                    _dict = v
                    for _kname in _kname_list:
                        _dict = {_kname: _dict}
                    trial_setup_dict = copy_merge(trial_setup_dict, _dict)
                else:
                    trial_setup_dict[k] = v
    elif mode in ["simple_train", "cont_train"]:
        trial_setup_dict = copy.deepcopy(optuna_setup_dict)

    trial_setup_dict['input_ndata_dim'] = len(trial_setup_dict['input_ndata_list'])
    trial_setup_dict['input_edata_dim'] = len(trial_setup_dict['input_edata_list'])
    devid_suffix = trial_setup_dict['devid_suffix']

    z = datetime.datetime.now()
    study_foldername = trial_setup_dict['study_foldername']
    if mode != "cont_train":
        modeltrained_foldername = "_{0}_{1}_{2:02d}_{3:02d}_s{4:02d}_ms{5:06d}_model".format(study_foldername, z.date(), z.hour, z.minute, z.second, z.microsecond).replace("-", "")
        trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername
    else:
        loss_mode = trial_setup_dict['loss_mode']
        modeltrained_foldername_prev = trial_setup_dict['modeltrained_foldername']
        trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername_prev + "##{}_CONT".format(loss_mode)
        modeltrained_foldername = trial_setup_dict['modeltrained_foldername']
    saving_dir = currentdir + "/" + modeltrained_foldername

    if trial_setup_dict.get('_test_mode') == "1":
        log_filename = "_TEST_trainlog"
    else:
        log_filename = modeltrained_foldername + "_trainlog"

    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    sys.stdout = open(log_filename, "w")

    if trial_setup_dict.get('training_seed') is None:
        trial_setup_dict['training_seed'] = random.randint(0, 9999)
    training_seed = trial_setup_dict['training_seed']

    print('trial_setup_dict')
    print(json.dumps(trial_setup_dict, indent=4))
    print('training_seed=', training_seed)

    trial_begin = time.time()
    _output_data_dict = _func.get_data(
        trial_setup_dict,
        batch_size=trial_setup_dict['batch_size'],
        test_split=trial_setup_dict['test_split'],
        vali_split=trial_setup_dict['vali_split'],
        currentdir=currentdir,
        return_indexes=True
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(_output_data_dict['train_features']).float(), torch.tensor(_output_data_dict['train_values']).float()), batch_size=trial_setup_dict['batch_size'], shuffle=True)
    vali_loader = DataLoader(TensorDataset(torch.tensor(_output_data_dict['vali_features']).float(), torch.tensor(_output_data_dict['vali_values']).float()), batch_size=trial_setup_dict['batch_size'], shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(_output_data_dict['test_features']).float(), torch.tensor(_output_data_dict['test_values']).float()), batch_size=trial_setup_dict['batch_size'], shuffle=False)

    node_num_info = _output_data_dict['node_num_info']
    train_vali_test_indexes = _output_data_dict['train_vali_test_indexes']

    if trial_setup_dict.get('node_num_info') is None:
        trial_setup_dict['node_num_info'] = node_num_info

    model = create_model(trial_setup_dict).to(_device)
    optimizer = create_optimizer(trial_setup_dict, model)
    loss_fn = nn.MSELoss() if trial_setup_dict['loss_mode'] == 'MAE' else nn.L1Loss()

    if mode == "cont_train":
        print("...Restoring checkpoint")
        checkpoint = torch.load(os.path.join(currentdir, modeltrained_foldername_prev, model
