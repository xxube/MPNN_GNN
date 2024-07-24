import os
import sys
import json
import time
import pickle
import datetime
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import optuna

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, currentdir + "/../../Model")
sys.path.insert(1, currentdir + "/../..")
_func = __import__("_func")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(trial_setup_dict, batch_size, vali_split, test_split):
    return _func.get_data(
        trial_setup_dict,
        batch_size=batch_size,
        vali_split=vali_split,
        test_split=test_split,
        currentdir=currentdir,
        return_indexes=True
    )

def create_model(trial_setup_dict):
    GNN_model = __import__(trial_setup_dict['modeltype_pyfilename'])
    model = GNN_model.MODEL(trial_setup_dict['model']).to(device)
    return model

def create_optimizer(trial_setup_dict, model_parameters):
    optimizer_chosen = trial_setup_dict["optimizer_chosen"]
    kwargs = trial_setup_dict[optimizer_chosen]
    optimizer = getattr(optim, optimizer_chosen)(model_parameters, **kwargs)
    return optimizer

def train_model(model, optimizer, train_dataset, loss_mode, feature_type="normal", training_seed=None):
    model.train()
    criterion = nn.L1Loss() if loss_mode == 'MAE' else nn.MSELoss()
    epoch_loss = 0
    total = 0

    if training_seed is not None:
        torch.manual_seed(training_seed)
    
    for batch, (features, labels) in enumerate(train_dataset):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += len(features)
        epoch_loss += loss.item() * len(features)
    
    return epoch_loss / total

def evaluate_model(model, dataset, loss_mode):
    model.eval()
    criterion = nn.L1Loss() if loss_mode == 'MAE' else nn.MSELoss()
    epoch_loss = 0
    total = 0

    with torch.no_grad():
        for batch, (features, labels) in enumerate(dataset):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total += len(features)
            epoch_loss += loss.item() * len(features)
    
    return epoch_loss / total

def calculate_r_square(model, dataset):
    model.eval()
    with torch.no_grad():
        features, labels = [], []
        for batch, (feat, lbl) in enumerate(dataset):
            features.append(feat.numpy())
            labels.append(lbl.numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        predictions = model(torch.tensor(features).to(device)).cpu().numpy()
        linreg = LinearRegression().fit(predictions.reshape(-1, 1), labels)
        linreg.coef_ = np.array([1])
        linreg.intercept_ = 0
        return linreg.score(predictions.reshape(-1, 1), labels)

def objective(trial, optuna_setup_dict, mode="trial"):
    if mode == "trial":
        for k, v in optuna_setup_dict.items():
            if isinstance(v, dict):
                if 'categorical' in v:
                    trial.suggest_categorical(k, v['categorical'])
                elif 'discrete_uniform' in v:
                    trial.suggest_discrete_uniform(k, v['discrete_uniform'][0], v['discrete_uniform'][1], v['discrete_uniform'][2])
                elif 'float' in v:
                    trial.suggest_float(k, v['float'][0], v['float'][1], step=v['float'][2].get('step'), log=v['float'][2].get('log', False))
                elif 'int' in v:
                    trial.suggest_int(k, v['int'][0], v['int'][1], step=v['int'][2].get('step'), log=v['int'][2].get('log', False))
                elif 'loguniform' in v:
                    trial.suggest_loguniform(k, v['loguniform'][0], v['loguniform'][1])
                elif 'uniform' in v:
                    trial.suggest_uniform(k, v['uniform'][0], v['uniform'][1])
            else:
                trial.set_user_attr(k, v)

        trial_setup_dict = trial.params
    else:
        trial_setup_dict = optuna_setup_dict

    trial_setup_dict['input_ndata_dim'] = len(trial_setup_dict['input_ndata_list'])
    trial_setup_dict['input_edata_dim'] = len(trial_setup_dict['input_edata_list'])

    data = get_data(trial_setup_dict, batch_size=trial_setup_dict['batch_size'], vali_split=trial_setup_dict['vali_split'], test_split=trial_setup_dict['test_split'])
    train_dataset = data['train_dataset']
    vali_dataset = data['vali_dataset']
    test_dataset = data['test_dataset']

    model = create_model(trial_setup_dict)
    optimizer = create_optimizer(trial_setup_dict, model.parameters())

    for epoch in range(trial_setup_dict['max_epoch_num']):
        train_loss = train_model(model, optimizer, train_dataset, loss_mode=trial_setup_dict['loss_mode'], training_seed=trial_setup_dict.get('training_seed'))
        vali_loss = evaluate_model(model, vali_dataset, loss_mode=trial_setup_dict['loss_mode'])
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss}, Vali Loss: {vali_loss}')

    if trial_setup_dict["eval_method"] == "r_square":
        return calculate_r_square(model, test_dataset)
    else:
        return evaluate_model(model, test_dataset, loss_mode=trial_setup_dict['loss_mode'])

def search(optuna_setup_dict, study):
    f = lambda y: objective(y, optuna_setup_dict)
    study_dir = os.path.join(currentdir, optuna_setup_dict["study_foldername"], "study")

    total_n_trials = optuna_setup_dict["study_total_n_trials"] - len(study.trials)
    num_per_batch = optuna_setup_dict["study_num_per_batch"]
    total_batch_count = math.ceil(total_n_trials / num_per_batch)

    for _ in range(1, total_batch_count + 1):
        n_trials = num_per_batch if num_per_batch < total_n_trials else total_n_trials
        if os.path.isfile(study_dir):
            with open(study_dir, 'rb') as _study_input:
                study = pickle.load(_study_input)
        study.optimize(f, n_trials=n_trials)
        pickle.dump(study, open(study_dir, "wb"))
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('new_filenames', type=str, nargs='+')
    parser.add_argument('--devid', type=int)
    parser.add_argument('--cont_study', type=str)
    parser.add_argument('--study_log', type=str)
    parser.add_argument('--test_mode', type=str)

    args = parser.parse_args()
    new_filenames = args.new_filenames
    devid = args.devid
    cont_study = args.cont_study
    study_log = args.study_log
    test_mode = args.test_mode

    if devid:
        devid_suffix = f"_#{devid}"
    else:
        devid_suffix = ""

    if cont_study:
        with open(os.path.join(currentdir, cont_study, "_setup.inp"), "r") as input_file:
            optuna_setup_dict = json.load(input_file)
        optuna_setup_dict["study_foldername"] = cont_study
        with open(os.path.join(currentdir, cont_study, "study"), 'rb') as _study_input:
            study = pickle.load(_study_input)
    else:
        setup_filename = new_filenames[0].replace(".inp", "")
        with open(setup_filename + ".inp", "r") as input_file:
            optuna_setup_dict = json.load(input_file)
        z = datetime.datetime.now()
        study_foldername = f"Optuna_StudyResult_{z.date()}_{z.hour:02d}{z.minute:02d}{devid_suffix}".replace("-", "")
        optuna_setup_dict["study_foldername"] = study_foldername
        study_dir = os.path.join(currentdir, study_foldername)
        os.makedirs(study_dir, exist_ok=True)
        with open(os.path.join(study_dir, "_setup.inp"), "w+") as output:
            output.write(json.dumps(optuna_setup_dict, indent=4))
        sampler_seed = random.randint(0, 9999) if optuna_setup_dict.get('sampler_seed') is None else optuna_setup_dict['sampler_seed']
        optuna_setup_dict['sampler_seed'] = sampler_seed
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction=optuna_setup_dict['study_direction'], pruner=optuna.pruners.HyperbandPruner(), sampler=sampler)

    optuna_setup_dict['devid_suffix'] = devid_suffix
    optuna_setup_dict['study_log_input'] = study_log
    optuna_setup_dict['study_log_output'] = f'study_log{len(glob.glob(study_dir + "/study_log*")) + 1}'
    optuna_setup_dict['_test_mode'] = test_mode

    search(optuna_setup_dict, study)

if __name__ == "__main__":
    main()
