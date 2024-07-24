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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dirname', type=str)
    parser.add_argument('_optuna_filename', type=str)
    parser.add_argument('--hide_plt', type=int)
    return parser.parse_args()

def load_trial_setup(checkdir, model_dirname):
    with open(os.path.join(checkdir, model_dirname, model_dirname + "_setting"), "r") as input_file:
        return json.load(input_file)

def import_optuna_module(_optuna_filename):
    if "/" in _optuna_filename:
        sys.path.insert(1, os.path.dirname(_optuna_filename))
        return __import__(os.path.basename(_optuna_filename))
    else:
        return __import__(_optuna_filename)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Restored from {checkpoint_path}")
        return epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None

def plot_results(x, y, label_list, hide_plt):
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
    ab = AnnotationBbox(offsetbox, (0, 0), xybox=xybox, xycoords='data', boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    ab.set_visible(False)
    fig.canvas.mpl_connect('motion_notify_event', hover)
    fig.set_size_inches(10.5, 9.5)

    linreg = LinearRegression(normalize=False, fit_intercept=True).fit(x.reshape(-1, 1), y)
    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    r_square = linreg.score(x.reshape(-1, 1), y)
    line_ = f'Regression line: y={linreg.intercept_:.2f}+{linreg.coef_[0]:.2f}x, r_square={r_square:.6f}'
    print(f"R^2 = {r_square:.6f}")

    x_coord, y_coord = x, y
    line = plt.scatter(x, y, s=30)
    ax.plot(x, linreg.intercept_ + linreg.coef_[0] * x, label=line_)
    leg = ax.legend(facecolor='white', loc='upper center')
    ax.add_artist(leg)

    if not hide_plt:
        plt.show()

    return r_square

def evaluate_model(model, features, values, tags, hide_plt):
    results = {}
    for data_type in ["train", "valid", "test"]:
        model.eval()
        with torch.no_grad():
            y_pred = model(features[data_type]).numpy().squeeze()
        y = values[data_type].numpy()
        tags[data_type] = [str(i) for i in tags[data_type].values]
        r_square = plot_results(np.array(y_pred), np.array(y), tags[data_type], hide_plt)
        results[data_type] = {"R_square": r_square, "MAE": nn.L1Loss()(torch.tensor(y), torch.tensor(y_pred)).item(),
                              "MAPE": (torch.mean(torch.abs((torch.tensor(y) - torch.tensor(y_pred)) / torch.tensor(y))) * 100).item(),
                              "RMS": torch.sqrt(nn.MSELoss()(torch.tensor(y), torch.tensor(y_pred))).item()}
        rms_log = torch.sqrt(nn.MSELoss()(torch.log10(torch.tensor(y)), torch.log10(torch.tensor(y_pred)))).item()
        deviation = 10 ** rms_log - 1
        print(f"deviation: {deviation}")
    return results

def save_evaluation_results(results, checkdir, model_dirname):
    with open(os.path.join(checkdir, model_dirname, "_eval_result"), "w") as output:
        json.dump(results, output, indent=4)

def main():
    args = parse_args()
    model_dirname = args.model_dirname
    _optuna_filename = args._optuna_filename.replace(".py", "")
    hide_plt = args.hide_plt

    currentdir = os.getcwd()
    checkdir = os.path.join(currentdir, '..')

    sys.path.insert(1, checkdir + "/../..")
    _func = __import__("_func")

    trial_setup_dict = load_trial_setup(checkdir, model_dirname)
    _optuna = import_optuna_module(_optuna_filename)

    _output_data_dict = _func.get_data(
        trial_setup_dict,
        batch_size=trial_setup_dict['batch_size'],
        vali_split=trial_setup_dict['vali_split'],
        test_split=trial_setup_dict['test_split'],
        currentdir=checkdir,
        as_dataset=False
    )

    train_features = torch.tensor(_output_data_dict['train_features'], dtype=torch.float32)
    vali_features = torch.tensor(_output_data_dict['vali_features'], dtype=torch.float32)
    test_features = torch.tensor(_output_data_dict['test_features'], dtype=torch.float32)
    train_values = torch.tensor(_output_data_dict['train_values'], dtype=torch.float32)
    vali_values = torch.tensor(_output_data_dict['vali_values'], dtype=torch.float32)
    test_values = torch.tensor(_output_data_dict['test_values'], dtype=torch.float32)
    train_tag = _output_data_dict['train_tag']
    vali_tag = _output_data_dict['vali_tag']
    test_tag = _output_data_dict['test_tag']

    model = _optuna.create_model(trial_setup_dict)
    optimizer = _optuna.create_optimizer(trial_setup_dict, model.parameters())

    load_checkpoint(model, optimizer, os.path.join(checkdir, model_dirname, model_dirname + '_checkpoint.pth'))

    features = {"train": train_features, "valid": vali_features, "test": test_features}
    values = {"train": train_values, "valid": vali_values, "test": test_values}
    tags = {"train": train_tag, "valid": vali_tag, "test": test_tag}

    results = evaluate_model(model, features, values, tags, hide_plt)
    save_evaluation_results(results, checkdir, model_dirname)

if __name__ == "__main__":
    main()
