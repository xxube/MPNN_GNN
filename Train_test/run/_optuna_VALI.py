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

# Helper function to get data
get_data = _func.get_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train(model, optimizer, train_loader, criterion):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def compute_r2_score(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_labels = np.array(all_labels)
    linreg = LinearRegression().fit(all_preds, all_labels)
    r2_score = linreg.score(all_preds, all_labels)
    return r2_score

def objective(trial, optuna_setup_dict, mode="trial"):
    print("Objective ...")

    if mode == "trial":
        trial_setup_dict = {}
        for k, v in optuna_setup_dict.items():
            if isinstance(v, dict):
                v_type = list(v.keys())[0]
                if v_type == 'categorical':
                    trial.suggest_categorical(k, v[v_type])
                elif v_type == 'discrete_uniform':
                    trial.suggest_discrete_uniform(k, v[v_type][0], v[v_type][1], v[v_type][2])
                elif v_type in ['float', 'int']:
                    trial.suggest_float(k, v[v_type][0], v[v_type][1], step=v.get('step'), log=v.get('log'))
                elif v_type in ['loguniform', 'uniform']:
                    trial.suggest_loguniform(k, v[v_type][0], v[v_type][1]) if v_type == 'loguniform' else trial.suggest_uniform(k, v[v_type][0], v[v_type][1])
            else:
                trial.set_user_attr(k, v)

        for _dict in [trial.user_attrs, trial.params]:
            for k, v in _dict.items():
                keys = k.split("|")
                keys.reverse()
                d = v
                for key in keys:
                    d = {key: d}
                trial_setup_dict = {**trial_setup_dict, **d}
    elif mode in ["simple_train", "cont_train"]:
        trial_setup_dict = copy.deepcopy(optuna_setup_dict)

    # Define saving directory
    z = datetime.datetime.now()
    study_foldername = trial_setup_dict['study_foldername']
    modeltrained_foldername = "_{0}_{1}_{2:02d}_{3:02d}_s{4:02d}_ms{5:06d}_model".format(study_foldername, z.date(), z.hour, z.minute, z.second, z.microsecond).replace("-", "")
    trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername
    saving_dir = os.path.join(currentdir, modeltrained_foldername)

    stdoutOrigin = sys.stdout
    log_filename = "_TEST_trainlog" if trial_setup_dict.get('_test_mode') == "1" else modeltrained_foldername + "_trainlog"
    sys.stdout = open(log_filename, "w")

    # Generate seed if not specified
    training_seed = trial_setup_dict.get('training_seed', random.randint(0, 9999))
    trial_setup_dict['training_seed'] = training_seed
    print('trial_setup_dict')
    print(json.dumps(trial_setup_dict, indent=4))
    print('training_seed=', training_seed)

    trial_begin = time.time()

    # Get train/test data
    _output_data_dict = _func.get_data(
        trial_setup_dict,
        batch_size=trial_setup_dict['batch_size'],
        test_split=trial_setup_dict['test_split'],
        vali_split=trial_setup_dict['vali_split'],
        currentdir=currentdir,
        return_indexes=True
    )
    train_loader = torch.utils.data.DataLoader(_output_data_dict['train_dataset'], batch_size=trial_setup_dict['batch_size'], shuffle=True)
    vali_loader = torch.utils.data.DataLoader(_output_data_dict['vali_dataset'], batch_size=trial_setup_dict['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(_output_data_dict['test_dataset'], batch_size=trial_setup_dict['batch_size'], shuffle=False)

    # Build model and optimizer
    model = create_model(trial_setup_dict)
    optimizer = create_optimizer(trial_setup_dict, model)

    if mode == "cont_train":
        print("...Restoring checkpoint")
        checkpoint = torch.load(os.path.join(saving_dir, 'checkpoint.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.MSELoss() if trial_setup_dict['loss_mode'] == 'MAE' else nn.L1Loss()

    best_vali_loss = float('inf')
    early_stopping_patience = trial_setup_dict.get('early_stopping_patience', 10)
    patience = 0

    for epoch in range(trial_setup_dict['max_epoch_num']):
        train_loss = train(model, optimizer, train_loader, criterion)
        vali_loss = evaluate(model, vali_loader, criterion)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {vali_loss:.4f}")

        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            patience = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(saving_dir, 'checkpoint.pth'))
        else:
            patience += 1

        if patience >= early_stopping_patience:
            print("Early stopping")
            break

    eval_result = compute_r2_score(model, test_loader)
    trial_end = time.time()

    sys.stdout.close()
    sys.stdout = stdoutOrigin
    with open(log_filename, "r") as input_file:
        _log = input_file.read()
    with open(os.path.join(saving_dir, "trainlog"), "w") as output_file:
        output_file.write(_log)
    os.remove(log_filename)

    return eval_result

def search(optuna_setup_dict, study):
    print(f"Sampler used is {study.sampler.__class__.__name__}")

    f = lambda y: objective(y, optuna_setup_dict)
    study_filename = optuna_setup_dict["study_foldername"]
    study_dir = os.path.join(currentdir, study_filename, "study")

    total_n_trials = optuna_setup_dict["study_total_n_trials"] - len(study.trials)
    num_per_batch = optuna_setup_dict["study_num_per_batch"]
    total_batch_count = math.ceil(total_n_trials / num_per_batch)

    record_study_log = optuna_setup_dict["study_log_input"]
    if record_study_log:
        study_log_inputdir = os.path.join(currentdir, optuna_setup_dict["study_log_input"])
        study_log_outputdir = os.path.join(currentdir, optuna_setup_dict["study_log_output"])

    for _i in range(1, total_batch_count + 1):
        n_trials = num_per_batch if _i * num_per_batch <= total_n_trials else total_n_trials % num_per_batch

        if os.path.isfile(study_dir):
            with open(study_dir, 'rb') as _study_input:
                study = pickle.load(_study_input)

        study.optimize(f, n_trials=n_trials)

        # save study
        pickle.dump(study, open(study_dir
