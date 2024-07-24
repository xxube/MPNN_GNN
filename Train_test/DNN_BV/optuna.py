import os
import sys
import json
import time
import random
import datetime
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pandas as pd
import numpy as np
import glob
import math
import copy

from sklearn.linear_model import LinearRegression

# Set current directory and insert paths for custom modules
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, current_directory + "/../../Model")
sys.path.insert(1, current_directory + "/../..")
func_module = __import__("_func")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to get data
get_data = func_module.get_data

# Model creation function
def initialize_model(config):
    print("Initializing Model ...")
    GNN_model = __import__(config['modeltype_pyfilename'])
    model = GNN_model.MODEL(config['model']).to(device)
    return model

# Optimizer creation function
def initialize_optimizer(config, model):
    print("Initializing Optimizer ...")
    optimizer_type = config["optimizer_chosen"]
    optimizer_params = config[optimizer_type]
    optimizer = getattr(optim, optimizer_type)(model.parameters(), **optimizer_params)
    return optimizer

# Training and evaluation function
def train_or_evaluate(model, optimizer, data_loader, loss_type, mode="eval", seed=None):
    if mode == "train" and seed is None:
        raise ValueError("ERROR: Training seed is required for training mode")

    criterion = nn.MSELoss() if loss_type == "MAE" else nn.L1Loss()
    
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_samples = 0

    for features, labels in data_loader:
        features, labels = features.to(device), labels.to(device)

        if mode == "train":
            optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        if mode == "train":
            loss.backward()
            optimizer.step()

        total_samples += labels.size(0)
        total_loss = (total_loss * (total_samples - labels.size(0)) + loss.item() * labels.size(0)) / total_samples

    if mode == "eval":
        print("Evaluating ...")

    return total_loss

# Evaluation function for different methods
def evaluate_model(model, optimizer, data_loader, config):
    method = config["eval_method"]

    if method == "default":
        loss_type = config['loss_mode']
        return train_or_evaluate(model, optimizer, data_loader, loss_type, "eval")
    elif method == "r_square":
        test_features = np.concatenate([x.numpy() for x, _ in data_loader])
        test_labels = np.concatenate([y.numpy() for _, y in data_loader])
        with torch.no_grad():
            predictions = model(torch.tensor(test_features).float().to(device)).cpu().numpy()
        
        predictions = np.array(predictions).reshape(-1, 1)
        real_values = np.array(test_labels)
        
        linreg = LinearRegression(normalize=False, fit_intercept=True).fit(predictions, real_values)
        linreg.coef_ = np.array([1])
        linreg.intercept_ = 0
        
        r_square = linreg.score(predictions, real_values)
        return r_square

# Objective function for Optuna
def objective(trial, optuna_config, mode="trial"):
    print("Starting Objective ...")

    if mode == "trial":
        for key, value in optuna_config.items():
            if isinstance(value, dict):
                param_type = list(value.keys())[0]
                if param_type == 'categorical':
                    choices = value[param_type]
                    trial.suggest_categorical(key, choices)
                elif param_type == 'discrete_uniform':
                    low, high, q = value[param_type]
                    trial.suggest_discrete_uniform(key, low, high, q)
                elif param_type in ['float', 'int']:
                    low, high = value[param_type][:2]
                    step = value[param_type][2].get('step') if len(value[param_type]) > 2 else None
                    log = value[param_type][2].get('log') if len(value[param_type]) > 2 else False
                    if step and log:
                        raise ValueError(f"ERROR: {key}, step and log cannot be used together")
                    if param_type == 'float':
                        trial.suggest_float(key, low, high, step=step, log=log)
                    elif param_type == 'int':
                        trial.suggest_int(key, low, high, step=step, log=log)
                elif param_type in ['loguniform', 'uniform']:
                    low, high = value[param_type]
                    if param_type == 'loguniform':
                        trial.suggest_loguniform(key, low, high)
                    elif param_type == 'uniform':
                        trial.suggest_uniform(key, low, high)
                else:
                    raise ValueError(f"ERROR: Unknown parameter type {key}, {value}")
            else:
                trial.set_user_attr(key, value)

        config = {}
        for dictionary in [trial.user_attrs, trial.params]:
            for key, value in dictionary.items():
                if "|" in key:
                    key_list = key.split("|")
                    key_list.reverse()
                    nested_dict = value
                    for sub_key in key_list:
                        nested_dict = {sub_key: nested_dict}
                    merge_dicts(config, nested_dict)
                else:
                    config[key] = value
    elif mode in ["simple_train", "cont_train"]:
        config = copy.deepcopy(optuna_config)

    config['input_ndata_dim'] = len(config['input_ndata_list'])
    config['input_edata_dim'] = len(config['input_edata_list'])

    z = datetime.datetime.now()
    study_foldername = config['study_foldername']
    if mode != "cont_train":
        model_foldername = f"_{study_foldername}_{z.date()}_{z.hour:02d}_{z.minute:02d}_s{z.second:02d}_ms{z.microsecond:06d}_model".replace("-", "")
        config['model_foldername'] = model_foldername
    else:
        loss_mode = config['loss_mode']
        previous_model_foldername = config['model_foldername']
        config['model_foldername'] = f"{previous_model_foldername}_##{loss_mode}_CONT"
        model_foldername = config['model_foldername']
    save_directory = os.path.join(current_directory, model_foldername)

    stdout_origin = sys.stdout
    log_filename = "_TEST_trainlog" if config.get('_test_mode') == "1" else f"{model_foldername}_trainlog"
    sys.stdout = open(log_filename, "w")

    if config.get('training_seed') is None:
        config['training_seed'] = random.randint(0, 9999)
    training_seed = config['training_seed']

    print('Config:')
    print(json.dumps(config, indent=4))
    print('Training Seed:', training_seed)

    trial_start_time = time.time()

    data = get_data(
        config,
        batch_size=config['batch_size'],
        test_split=config['test_split'],
        vali_split=config['vali_split'],
        currentdir=current_directory,
        return_indexes=True
    )
    dummy_train_data = data['stage1_train_dataset']
    train_data = data['train_dataset']
    dummy_vali_data = data['stage1_vali_dataset']
    vali_data = data['vali_dataset']
    dummy_test_data = data['stage1_test_dataset']
    test_data = data['test_dataset']

    node_info = data['node_num_info']
    data_indexes = data['train_vali_test_indexes']

    if config.get('node_num_info') is None:
        config['node_num_info'] = node_info

    model = initialize_model(config)
    optimizer = initialize_optimizer(config, model)

    if mode == "cont_train":
        print("Restoring checkpoint ...")
        checkpoint = torch.load(os.path.join(current_directory, previous_model_foldername, previous_model_foldername, "model.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

    stage1_cutoff = config['stage1_cutoff']
    stage1_skip = config.get('stage1_skip')
    stage2_converge = config.get('stage2_converge', 0.0)
    stage2_fluctuate_dv = config['stage2_fluctuate_dv']
    stage2_fluctuate_pc = config['stage2_fluctuate_pc']
    loss_mode = config['loss_mode']
    # Additional logic for training and evaluation can be added here

if __name__ == "__main__":
    # Argument parsing and other initial setups can be added here
    pass
