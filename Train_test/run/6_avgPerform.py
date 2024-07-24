import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_filename', type=str)
    parser.add_argument('--hide_plt', type=int, default=0)
    parser.add_argument('--round_digits', type=int, default=2)
    args = parser.parse_args()
    args.result_filename = args.result_filename.replace('.csv', '') + '.csv'
    return args

def plot_model(x, y, _entry, round_digits, saving_dir, hide_plt):
    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    linreg = LinearRegression().fit(x.reshape(-1, 1), y)
    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    R_square = linreg.score(x.reshape(-1, 1), y)

    plt.scatter(x, y, s=10)

    _plot_min = 0
    _plot_max = 50
    ax.plot([_plot_min - 1, _plot_max + 1], [_plot_min - 1, _plot_max + 1], 'k--', lw=1.75)
    plt.xlim(_plot_min, _plot_max)
    plt.ylim(_plot_min, _plot_max)

    plt.xlabel('Experimental rate constant\n k (1000/h⁻¹)', fontsize=36)
    plt.ylabel('Predicted rate constant\n k (1000/h⁻¹)', fontsize=36)

    legend_text = f'R²    = {round(R_square, round_digits) if R_square >= 0 else "< 0"}'
    legend_text += f'\nMAE = {round(mean_absolute_error(y, x), round_digits)}'
    legend_text += f'\nRMS = {round(np.sqrt(mean_squared_error(y, x)), round_digits)}'
    at = AnchoredText(legend_text, prop=dict(size=36), frameon=True, loc='lower right')
    ax.add_artist(at)

    fig.text(-0.05, 0.95, {'train': 'a)', 'vali': 'b)', 'test': 'c)'}[_entry],
             horizontalalignment='left', verticalalignment='top', size=45)
    plt.title({'train': 'Training Set Performance', 'vali': 'Validation Set Performance', 'test': 'Testing Set Performance'}[_entry],
              size=35, pad=25)

    plt.savefig(os.path.join(saving_dir, f'{_entry}.png'), bbox_inches='tight', dpi=500)

    if not hide_plt:
        plt.show()

    return R_square

def plotall_model(xdict, ydict, saving_dir, hide_plt):
    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    scatter_size = 15
    for k, x in xdict.items():
        y = ydict[k]
        if k == "train":
            train_scat = plt.scatter(x, y, s=scatter_size, c='red', marker='o')
        elif k == "vali":
            vali_scat = plt.scatter(x, y, s=scatter_size, c='green', marker='x')
        elif k == "test":
            test_scat = plt.scatter(x, y, s=scatter_size, c='blue', marker='^')
    lgnd = plt.legend((train_scat, vali_scat, test_scat),
                      ('Training set', 'Validation set', 'Testing set'),
                      scatterpoints=1,
                      loc='lower right',
                      fontsize=20)
    for i in range(3):
        lgnd.legendHandles[i]._sizes = [50]

    _plot_min = 0
    _plot_max = 50
    ax.plot([_plot_min - 1, _plot_max + 1], [_plot_min - 1, _plot_max + 1], 'k--', lw=1.75)

    plt.xlabel("Experimental rate constant\n k (1000/h⁻¹)", fontsize=36)
    plt.ylabel("Predicted rate constant\n k (1000/h⁻¹)", fontsize=36)

    plt.xlim(_plot_min, _plot_max)
    plt.ylim(_plot_min, _plot_max)

    title = 'Performance of \nTraining/Validation/Testing Sets'
    plt.title(title, size=35, pad=25)

    plt.savefig(os.path.join(saving_dir, "ALL.png"), bbox_inches="tight", dpi=450)

    if not hide_plt:
        plt.show()

def main():
    args = parse_arguments()

    currentdir = os.getcwd()
    saving_dir = os.path.join(currentdir, 'avgPerform')
    os.makedirs(saving_dir, exist_ok=True)

    input_df = pd.read_csv(os.path.join(currentdir, args.result_filename))

    result = {entry: {'values': input_df[input_df['data_type'] == entry]['y_true'].values.tolist(),
                      'pred': input_df[input_df['data_type'] == entry]['y_pred'].values.tolist()}
              for entry in ['train', 'vali', 'test']}

    all_xdict, all_ydict = {}, {}
    for entry in ['train', 'vali', 'test']:
        y = np.array(result[entry]['values'])
        y_pred = np.array(result[entry]['pred'])

        result[entry]['MAE'] = mean_absolute_error(y, y_pred)
        result[entry]['RMS'] = np.sqrt(mean_squared_error(y, y_pred))
        result[entry]['R_square'] = plot_model(y, y_pred, entry, args.round_digits, saving_dir, args.hide_plt)

        all_xdict[entry] = y
        all_ydict[entry] = y_pred

    plotall_model(all_xdict, all_ydict, saving_dir, args.hide_plt)

    with open(os.path.join(saving_dir, 'avgPerform_LOG'), 'w') as output:
        json.dump(result, output, indent=4)

if __name__ == "__main__":
    main()
