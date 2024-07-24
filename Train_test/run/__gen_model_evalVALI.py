import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import argparse
import warnings
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox
from sklearn.linear_model import LinearRegression
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('model_dirname', type=str)
parser.add_argument('_optuna_filename', type=str)
parser.add_argument('--hide_plt', type=int)

args = parser.parse_args()
model_dirname = args.model_dirname
_optuna_filename = args._optuna_filename.replace(".py", "")
hide_plt = args.hide_plt

currentdir = os.getcwd()
checkdir = os.path.join(currentdir, '..')

sys.path.insert(1, checkdir + "/../..")
_func = __import__("_func")

with open(os.path.join(checkdir, model_dirname, model_dirname + "_setting"), "r") as input_file:
    trial_setup_dict = json.load(input_file)

if "/" in _optuna_filename:
    sys.path.insert(1, os.path.dirname(_optuna_filename))
    _optuna = __import__(os.path.basename(_optuna_filename))
else:
    _optuna = __import__(_optuna_filename)

# Get train/test data
_output_data_dict = _func.get_data(
    trial_setup_dict,
    batch_size=trial_setup_dict['batch_size'],
    vali_split=trial_setup_dict['vali_split'],
    test_split=trial_setup_dict['test_split'],
    currentdir=checkdir,
    as_dataset=False
)

train_tag = _output_data_dict['train_tag']
train_features = torch.tensor(_output_data_dict['train_features'], dtype=torch.float32)
train_values = torch.tensor(_output_data_dict['train_values'], dtype=torch.float32)

vali_tag = _output_data_dict['vali_tag']
vali_features = torch.tensor(_output_data_dict['vali_features'], dtype=torch.float32)
vali_values = torch.tensor(_output_data_dict['vali_values'], dtype=torch.float32)

test_tag = _output_data_dict['test_tag']
test_features = torch.tensor(_output_data_dict['test_features'], dtype=torch.float32)
test_values = torch.tensor(_output_data_dict['test_values'], dtype=torch.float32)

# Define model and optimizer
model = _optuna.create_model(trial_setup_dict)
optimizer = _optuna.create_optimizer(trial_setup_dict, model.parameters())

checkpoint_path = os.path.join(checkdir, model_dirname, model_dirname + '_checkpoint.pth')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Restored from {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}")

# Function to plot model results
def plot_model(x, y, label_list):
    def hover(event):
        if line.contains(event)[0]:
            data_pts = line.contains(event)[1]["ind"]
            ind = data_pts[0]
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab.set_visible(True)
            ab.xy = (x_coord[ind], y_coord[ind])
            plt.annotate(label_list[ind], (x_coord[ind], y_coord[ind]), color='red', size=15)
        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()

    offsetbox = TextArea("-----")

    fig, ax = plt.subplots()
    xybox = (90., 90.)
    ab = AnnotationBbox(offsetbox, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    ab.set_visible(False)
    fig.canvas.mpl_connect('motion_notify_event', hover)
    fig = plt.gcf()
    fig.set_size_inches(10.5, 9.5)

    linreg = LinearRegression(normalize=False, fit_intercept=True).fit(x.reshape(-1, 1), y)
    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    slope = float(linreg.coef_)
    intercept = float(linreg.intercept_)
    r_square_after = linreg.score(x.reshape(-1, 1), y)
    line_ = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r_square={r_square_after:.6f}'

    print(f"R^2_after = {r_square_after:.6f}")

    x_coord = x
    y_coord = y

    line = plt.scatter(x, y, s=30)

    ax.plot(x, intercept + slope * x, label=line_)
    leg = ax.legend(facecolor='white', loc='upper center')

    ax.add_artist(leg)
    if not hide_plt:
        plt.show()

    return r_square_after

# Model evaluation and result storage
_result = {"train": {}, "valid": {}, "test": {}}

_features = {"train": train_features, "valid": vali_features, "test": test_features}
_values = {"train": train_values, "valid": vali_values, "test": test_values}
_tag = {"train": train_tag, "valid": vali_tag, "test": test_tag}

for _entry in ["train", "valid", "test"]:
    model.eval()
    with torch.no_grad():
        y_pred = model(_features[_entry]).numpy().squeeze()
    y = _values[_entry].numpy()
    _tag[_entry] = [str(i) for i in _tag[_entry].values]
    R_square = plot_model(np.array(y_pred), np.array(y), _tag[_entry])
    _result[_entry]['R_square'] = R_square
    MAE = nn.L1Loss()
    MAPE = lambda y_true, y_pred: torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    RMS = nn.MSELoss()
    print("MAE", MAE(torch.tensor(y), torch.tensor(y_pred)).item())
    _result[_entry]["MAE"] = float(MAE(torch.tensor(y), torch.tensor(y_pred)).item())
    print("MAPE", MAPE(torch.tensor(y), torch.tensor(y_pred)).item())
    _result[_entry]["MAPE"] = float(MAPE(torch.tensor(y), torch.tensor(y_pred)).item())
    print("RMS", torch.sqrt(RMS(torch.tensor(y), torch.tensor(y_pred))).item())
    _result[_entry]["RMS"] = float(torch.sqrt(RMS(torch.tensor(y), torch.tensor(y_pred))).item())

    RMS_log = nn.MSELoss()
    RMS_log_result = torch.sqrt(RMS_log(torch.log10(torch.tensor(y)), torch.log10(torch.tensor(y_pred)))).item()
    deviation = 10 ** RMS_log_result - 1
    print("deviation", deviation)

with open(os.path.join(checkdir, model_dirname, "_eval_result"), "w") as output:
    output.write(json.dumps(_result, indent=4))
