import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pandas as pd
import numpy as np
import os
import time
import pickle
import datetime
import json
import copy
import math
import argparse
import glob
import random
import sys

from sklearn.linear_model import LinearRegression

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, currentdir + "/../../Model")
sys.path.insert(1, currentdir + "/../..")
_func = __import__("_func")

# Get CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to get data
get_data = _func.get_data

# Model defining
def create_model(trial_setup_dict):
    print("Model ...")
    GNN_model = __import__(trial_setup_dict['modeltype_pyfilename'])
    model = GNN_model.MODEL(trial_setup_dict['model']).to(device)
    return model

def create_optimizer(trial_setup_dict, model):
    print("Optimizer ...")
    optimizer_chosen = trial_setup_dict["optimizer_chosen"]
    kwargs = trial_setup_dict[optimizer_chosen]
    optimizer = getattr(optim, optimizer_chosen)(model.parameters(), **kwargs)
    return optimizer

def learn(model, optimizer, train_loader, loss_mode, mode="eval", training_seed=None):
    if mode == "train" and training_seed is None:
        raise ValueError("ERROR, require to set a training seed")

    criterion = nn.MSELoss() if loss_mode == "MAE" else nn.L1Loss()

    if mode == "train":
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        if mode == "train":
            optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        if mode == "train":
            loss.backward()
            optimizer.step()

        total += labels.size(0)
        epoch_loss = (epoch_loss * (total - labels.size(0)) + loss.item() * labels.size(0)) / total

    if mode == "eval":
        print("Eval ...")

    return epoch_loss

def evaluate(model, optimizer, test_loader, trial_setup_dict):
    method = trial_setup_dict["eval_method"]

    if method == "default":
        loss_mode = trial_setup_dict['loss_mode']
        return learn(model, optimizer, test_loader, loss_mode, "eval_silent")
    elif method == "r_square":
        # Pretreatment
        test_features = np.concatenate([x.numpy() for x, _ in test_loader])
        test_values = np.concatenate([y.numpy() for _, y in test_loader])
        with torch.no_grad():
            pred = model(torch.tensor(test_features).float().to(device)).cpu().numpy()
        real = test_values
        _pred = np.array(pred).reshape(-1, 1)
        _real = np.array(real)
        linreg = LinearRegression(normalize=False, fit_intercept=True).fit(_pred, _real)
        linreg.coef_ = np.array([1])
        linreg.intercept_ = 0
        get_r_square = linreg.score(_pred, _real)
        return get_r_square

def objective(trial, optuna_setup_dict, mode="trial"):
    print("Objective ...")

    if mode == "trial":
        # Initialize trial
        for k, v in optuna_setup_dict.items():
            if isinstance(v, dict):
                v_type = list(v.keys())[0]
                if v_type == 'categorical':
                    v_choices = v[v_type]
                    trial.suggest_categorical(k, v_choices)
                elif v_type == 'discrete_uniform':
                    v_low, v_high, v_q = v[v_type]
                    trial.suggest_discrete_uniform(k, v_low, v_high, v_q)
                elif v_type in ['float', 'int']:
                    v_low, v_high = v[v_type][:2]
                    v_step = v[v_type][2].get('step') if len(v[v_type]) > 2 else None
                    v_log = v[v_type][2].get('log') if len(v[v_type]) > 2 else False
                    if v_step and v_log:
                        raise ValueError(f"ERROR at {k}, step and log cannot be used together")
                    if v_type == 'float':
                        trial.suggest_float(k, v_low, v_high, step=v_step, log=v_log)
                    elif v_type == 'int':
                        trial.suggest_int(k, v_low, v_high, step=v_step, log=v_log)
                elif v_type in ['loguniform', 'uniform']:
                    v_low, v_high = v[v_type]
                    if v_type == 'loguniform':
                        trial.suggest_loguniform(k, v_low, v_high)
                    elif v_type == 'uniform':
                        trial.suggest_uniform(k, v_low, v_high)
                else:
                    raise ValueError(f"ERROR v_type {k}, {v}")
            else:
                trial.set_user_attr(k, v)

        trial_setup_dict = {}
        for _dict in [trial.user_attrs, trial.params]:
            for k, v in _dict.items():
                if "|" in k:
                    _kname_list = k.split("|")
                    _kname_list.reverse()
                    _dict = v
                    for _kname in _kname_list:
                        _dict = {_kname: _dict}
                    merge_dicts(trial_setup_dict, _dict)
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
        modeltrained_foldername = f"_{study_foldername}_{z.date()}_{z.hour:02d}_{z.minute:02d}_s{z.second:02d}_ms{z.microsecond:06d}_model".replace("-", "")
        trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername
    else:
        loss_mode = trial_setup_dict['loss_mode']
        modeltrained_foldername_prev = trial_setup_dict['modeltrained_foldername']
        trial_setup_dict['modeltrained_foldername'] = f"{modeltrained_foldername_prev}_##{loss_mode}_CONT"
        modeltrained_foldername = trial_setup_dict['modeltrained_foldername']
    saving_dir = os.path.join(currentdir, modeltrained_foldername)

    stdoutOrigin = sys.stdout
    log_filename = "_TEST_trainlog" if trial_setup_dict.get('_test_mode') == "1" else f"{modeltrained_foldername}_trainlog"
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
    dummy_train_dataset = _output_data_dict['stage1_train_dataset']
    train_dataset = _output_data_dict['train_dataset']
    dummy_vali_dataset = _output_data_dict['stage1_vali_dataset']
    vali_dataset = _output_data_dict['vali_dataset']
    dummy_test_dataset = _output_data_dict['stage1_test_dataset']
    test_dataset = _output_data_dict['test_dataset']

    node_num_info = _output_data_dict['node_num_info']
    train_vali_test_indexes = _output_data_dict['train_vali_test_indexes']

    if trial_setup_dict.get('node_num_info') is None:
        trial_setup_dict['node_num_info'] = node_num_info

    model = create_model(trial_setup_dict)
    optimizer = create_optimizer(trial_setup_dict, model)

    if mode == "cont_train":
        print("...Restoring checkpoint")
        checkpoint = torch.load(os.path.join(currentdir, modeltrained_foldername_prev, modeltrained_foldername_prev, "model.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

    stage1_cutoff = trial_setup_dict['stage1_cutoff']
    stage1_skip = trial_setup_dict.get('stage1_skip')
    stage2_converge = trial_setup_dict.get('stage2_converge', 0.0)
    stage2_fluctuate_dv = trial_setup_dict['stage2_fluctuate_dv']
    stage2_fluctuate_pc = trial_setup_dict['stage2_fluctuate_pc']
    loss_mode
