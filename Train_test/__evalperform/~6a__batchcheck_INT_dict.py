import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from statistics import mean
import pickle
import argparse

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--modellist', type=str)
parser.add_argument('--include_CONT', type=int)
parser.add_argument('--mode', type=str)

args = parser.parse_args()
_modellist = args.modellist
_include_CONT = args.include_CONT
_mode = args.mode

currentdir = os.getcwd()

# Prepare model folder list
if _modellist is None:
    model_folder_list = [model_folder for model_folder in os.listdir(currentdir)
                         if "_model" in model_folder and model_folder[0] != "~"
                         and os.path.isdir(os.path.join(currentdir, model_folder))]
else:
    with open(os.path.join(currentdir, _modellist), "r") as input_file:
        model_folder_list = input_file.read().split("\n")
        model_folder_list = [x for x in model_folder_list if x]

if not _include_CONT:
    model_folder_list = [x for x in model_folder_list if "_CONT" not in x]

# Initialize output dictionary
_output_dict = {
    "lig": {},
    "rctXlig": {}
}

_sample = model_folder_list[0]
_sample_df = pd.read_csv(os.path.join(currentdir, _sample, '_INT_dict.csv'))

lig_id_list = sorted(list(set(_sample_df['lig_id'].values.tolist())))
rct_id_list = sorted(list(set(_sample_df['rct_id'].values.tolist())))

# Helper function to detect outliers
def detect_outlier(data):
    q1, q3 = np.percentile(sorted(data), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = [(idx, x) for idx, x in enumerate(data) if x <= lower_bound or x >= upper_bound]
    return [x[0] for x in outliers], [x[1] for x in outliers]

# Initialize output dictionary structure
for lig_id in lig_id_list:
    _output_dict["lig"][lig_id] = []
    _output_dict["rctXlig"][lig_id] = {}
    for rct_id in rct_id_list:
        _output_dict["rctXlig"][lig_id][rct_id] = []

# Aggregate data from model folders
for model_folder in model_folder_list:
    _INT_df = pd.read_csv(os.path.join(currentdir, model_folder, '_INT_dict.csv'))
    for lig_id in lig_id_list:
        lig_E_value = _INT_df.loc[_INT_df['lig_id'] == lig_id]['inputs_E2_Weighted'].values[0]
        _output_dict["lig"][lig_id].append(lig_E_value)
        for rct_id in rct_id_list:
            rctXlig_E_value = _INT_df.loc[(_INT_df['lig_id'] == lig_id) & (_INT_df['rct_id'] == rct_id)]['inputs_E1_Weighted'].values[0]
            _output_dict["rctXlig"][lig_id][rct_id].append(rctXlig_E_value)

# Create saving directory if it does not exist
saving_dir = os.path.join(currentdir, "batchcheck_INT")
os.makedirs(saving_dir, exist_ok=True)

# Plotting
ligE_avg_list = [mean(_output_dict["lig"][k]) for k in _output_dict["lig"].keys()]
lig_id_list_sorted = np.argsort(ligE_avg_list) + 1

plot_set = [
    (lig_id_list, ""),
    (lig_id_list_sorted, "_sorted"),
]

_plot_dict = {}

for plot_id_list, _suffix in plot_set:
    x_labels = [str(i) for i in plot_id_list]
    data_set = [_output_dict["lig"][lig_id] for lig_id in plot_id_list]
    fig = plt.figure(figsize=(10, 7))
    top = int(max([i for i in max(data_set)]) + 1)
    bottom = 0

    if _mode is None:
        plt.plot(data_set, 'x')
        plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
        plt.xticks(range(len(x_labels)), x_labels, fontsize=20)
    elif _mode == "boxplot":
        plt.boxplot(data_set)
        plt.xticks(range(1, len(x_labels) + 1), x_labels, fontsize=20)

    plt.gca().yaxis.set_major_formatter(plt.StrMethodFormatter('{x:,.1f}'))
    plt.rc('ytick', labelsize=20)
    title = "ligE{}".format(_suffix)
    plt.ylim(top=top, bottom=bottom)
    plt.xlabel("Ligands", fontsize=26)

    html_str = mpld3.fig_to_html(fig)
    with open(os.path.join(saving_dir, "_INT_lig{}.html".format(_suffix)), "w") as html_file:
        html_file.write(html_str)

    _plot_dict[title] = {
        "dataset": data_set,
        "order": plot_id_list,
    }
    plt.savefig(os.path.join(saving_dir, "_INT_lig{}.png".format(_suffix)))

# Reactant vs Ligand Plot for a specific ligand
def plot_rctXlig(lig_id, rct_id_list, _output_dict, saving_dir, _mode, model_folder_list):
    rctXligE_avg_list = {lig_id: {rct_id: mean(_output_dict["rctXlig"][lig_id][rct_id]) for rct_id in rct_id_list}}
    y_unsorted = [rctXligE_avg_list[lig_id][rct_id] for rct_id in rct_id_list]
    plot_id_list = [x for _, x in sorted(zip(y_unsorted, rct_id_list))]
    data_set = [_output_dict["rctXlig"][lig_id][rct_id] for rct_id in plot_id_list]

    fig = plt.figure(figsize=(10, 7))
    top = (int(max([i for i in max(data_set)]) * 2) + 1) / 2
    bottom = int(min([i for i in min(data_set)]) * 2) / 2

    if _mode is None:
        plt.plot(data_set, 'x')
        plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(len(plot_id_list)), plot_id_list, fontsize=20)
    elif _mode == "boxplot":
        plt.boxplot(data_set)
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(1, len(plot_id_list) + 1), plot_id_list, fontsize=20)

    plt.ylim(top=top, bottom=bottom)
    plt.rc('ytick', labelsize=20)
    title = "rctXlig of ligand {}".format(lig_id)
    plt.title(title)
    plt.xlabel("Reactants", fontsize=26)

    html_str = mpld3.fig_to_html(fig)
    with open(os.path.join(saving_dir, "_INT_rctXligE_lig{}.html".format(lig_id)), "w") as html_file:
        html_file.write(html_str)

    plt.savefig(os.path.join(saving_dir, "_INT_rctXligE_lig{}.png".format(lig_id)))

plot_rctXlig(7, rct_id_list, _output_dict, saving_dir, _mode, model_folder_list)

# Reactant vs Ligand Plot for a specific reactant
def plot_ligXrct(current_rct_id, lig_id_list, _output_dict, saving_dir, _mode, model_folder_list):
    rctXligE_avg_list = {lig_id: {current_rct_id: mean(_output_dict["rctXlig"][lig_id][current_rct_id]) for lig_id in lig_id_list}}
    y_unsorted = [rctXligE_avg_list[lig_id][current_rct_id] for lig_id in lig_id_list]
    plot_id_list = [x for _, x in sorted(zip(y_unsorted, lig_id_list))]
    data_set = [_output_dict["rctXlig"][lig_id][current_rct_id] for lig_id in plot_id_list]

    fig = plt.figure(figsize=(10, 7))
    top = (int(max([i for i in max(data_set)]) * 2) + 1) / 2
    bottom = int(min([i for i in min(data_set)]) * 2) / 2

    if _mode is None:
        plt.plot(data_set, 'x')
        plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(len(plot_id_list)), plot_id_list, fontsize=20)
    elif _mode == "boxplot":
        plt.boxplot(data_set)
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(1, len(plot_id_list) + 1), plot_id_list, fontsize=20)

    plt.rc('ytick', labelsize=20)
    title = "rctXligE_rct{}".format(current_rct_id)
    plt.ylim(top=top, bottom=bottom)
    plt.title(title)
    plt.xlabel("Ligands", fontsize=26)

    html_str = mpld3.fig_to_html(fig)
    with open(os.path.join(saving_dir, "_INT_rctXligE_rct{}.html".format(current_rct_id)), "w") as html_file:
        html_file.write(html_str)

    plt.savefig(os.path.join(saving_dir, "_INT_rctXligE_rct{}.png".format(current_rct_id)))

plot_ligXrct(1, lig_id_list, _output_dict, saving_dir, _mode, model_folder_list)

# Save plot dictionary
with open(os.path.join(saving_dir, '_INT_lig.pickle'), 'wb') as handle:
    pickle.dump(_plot_dict, handle)
