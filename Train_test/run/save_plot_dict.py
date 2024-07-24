import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from statistics import mean
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modellist', type=str, default=None)
    parser.add_argument('--include_CONT', type=int, default=0)
    parser.add_argument('--mode', type=str, default=None)
    return parser.parse_args()

def get_model_folders(currentdir, modellist, include_CONT):
    if modellist is None:
        return [folder for folder in os.listdir(currentdir)
                if "_model" in folder and folder[0] != "~" and os.path.isdir(os.path.join(currentdir, folder))]
    else:
        with open(os.path.join(currentdir, modellist), "r") as input_file:
            model_folder_list = input_file.read().split("\n")
            return [x for x in model_folder_list if x]

def filter_cont_models(model_folder_list, include_CONT):
    if not include_CONT:
        return [x for x in model_folder_list if "_CONT" not in x]
    return model_folder_list

def initialize_output_dict(lig_id_list, rct_id_list):
    output_dict = {
        "lig": {lig_id: [] for lig_id in lig_id_list},
        "rctXlig": {lig_id: {rct_id: [] for rct_id in rct_id_list} for lig_id in lig_id_list}
    }
    return output_dict

def detect_outliers(data):
    q1, q3 = np.percentile(sorted(data), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return [(idx, x) for idx, x in enumerate(data) if x <= lower_bound or x >= upper_bound]

def aggregate_data(model_folder_list, lig_id_list, rct_id_list, currentdir, output_dict):
    for model_folder in model_folder_list:
        _INT_df = pd.read_csv(os.path.join(currentdir, model_folder, '_INT_dict.csv'))
        for lig_id in lig_id_list:
            lig_E_value = _INT_df.loc[_INT_df['lig_id'] == lig_id]['inputs_E2_Weighted'].values[0]
            output_dict["lig"][lig_id].append(lig_E_value)
            for rct_id in rct_id_list:
                rctXlig_E_value = _INT_df.loc[(_INT_df['lig_id'] == lig_id) & (_INT_df['rct_id'] == rct_id)]['inputs_E1_Weighted'].values[0]
                output_dict["rctXlig"][lig_id][rct_id].append(rctXlig_E_value)

def create_saving_directory(saving_dir):
    os.makedirs(saving_dir, exist_ok=True)

def plot_ligands(output_dict, model_folder_list, lig_id_list, saving_dir, mode):
    ligE_avg_list = [mean(output_dict["lig"][k]) for k in output_dict["lig"].keys()]
    lig_id_list_sorted = np.argsort(ligE_avg_list) + 1

    plot_set = [
        (lig_id_list, ""),
        (lig_id_list_sorted, "_sorted"),
    ]

    plot_dict = {}

    for plot_id_list, suffix in plot_set:
        x_labels = [str(i) for i in plot_id_list]
        data_set = [output_dict["lig"][lig_id] for lig_id in plot_id_list]
        fig = plt.figure(figsize=(10, 7))
        top = int(max([i for i in max(data_set)]) + 1)
        bottom = 0

        if mode is None:
            plt.plot(data_set, 'x')
            plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
            plt.xticks(range(len(x_labels)), x_labels, fontsize=20)
        elif mode == "boxplot":
            plt.boxplot(data_set)
            plt.xticks(range(1, len(x_labels) + 1), x_labels, fontsize=20)

        plt.gca().yaxis.set_major_formatter(plt.StrMethodFormatter('{x:,.1f}'))
        plt.rc('ytick', labelsize=20)
        title = "ligE{}".format(suffix)
        plt.ylim(top=top, bottom=bottom)
        plt.xlabel("Ligands", fontsize=26)

        html_str = mpld3.fig_to_html(fig)
        with open(os.path.join(saving_dir, "_INT_lig{}.html".format(suffix)), "w") as html_file:
            html_file.write(html_str)

        plot_dict[title] = {
            "dataset": data_set,
            "order": plot_id_list,
        }
        plt.savefig(os.path.join(saving_dir, "_INT_lig{}.png".format(suffix)))

    return plot_dict

def plot_reactants_vs_ligand(lig_id, rct_id_list, output_dict, saving_dir, mode, model_folder_list):
    rctXligE_avg_list = {lig_id: {rct_id: mean(output_dict["rctXlig"][lig_id][rct_id]) for rct_id in rct_id_list}}
    y_unsorted = [rctXligE_avg_list[lig_id][rct_id] for rct_id in rct_id_list]
    plot_id_list = [x for _, x in sorted(zip(y_unsorted, rct_id_list))]
    data_set = [output_dict["rctXlig"][lig_id][rct_id] for rct_id in plot_id_list]

    fig = plt.figure(figsize=(10, 7))
    top = (int(max([i for i in max(data_set)]) * 2) + 1) / 2
    bottom = int(min([i for i in min(data_set)]) * 2) / 2

    if mode is None:
        plt.plot(data_set, 'x')
        plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(len(plot_id_list)), plot_id_list, fontsize=20)
    elif mode == "boxplot":
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

def plot_ligands_vs_reactant(rct_id, lig_id_list, output_dict, saving_dir, mode, model_folder_list):
    rctXligE_avg_list = {lig_id: {rct_id: mean(output_dict["rctXlig"][lig_id][rct_id]) for lig_id in lig_id_list}}
    y_unsorted = [rctXligE_avg_list[lig_id][rct_id] for lig_id in lig_id_list]
    plot_id_list = [x for _, x in sorted(zip(y_unsorted, lig_id_list))]
    data_set = [output_dict["rctXlig"][lig_id][rct_id] for lig_id in plot_id_list]

    fig = plt.figure(figsize=(10, 7))
    top = (int(max([i for i in max(data_set)]) * 2) + 1) / 2
    bottom = int(min([i for i in min(data_set)]) * 2) / 2

    if mode is None:
        plt.plot(data_set, 'x')
        plt.legend([i + 1 for i in range(len(model_folder_list))], loc="upper right")
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(len(plot_id_list)), plot_id_list, fontsize=20)
    elif mode == "boxplot":
        plt.boxplot(data_set)
        plt.yticks(np.arange(bottom, top + 0.5, 0.5))
        plt.xticks(range(1, len(plot_id_list) + 1), plot_id_list, fontsize=20)

    plt.rc('ytick', labelsize=20)
    title = "rctXligE_rct{}".format(rct_id)
    plt.ylim(top=top, bottom=bottom)
    plt.title(title)
    plt.xlabel("Ligands", fontsize=26)

    html_str = mpld3.fig_to_html(fig)
    with open(os.path.join(saving_dir, "_INT_rctXligE_rct{}.html".format(rct_id)), "w") as html_file:
        html_file.write(html_str)

    plt.savefig(os.path.join(saving_dir, "_INT_rctXligE_rct{}.png".format(rct_id)))

def save_plot_dict(plot_dict, saving_dir):
    with open(os.path.join(saving_dir, '_INT_lig.pickle'), 'wb') as handle:
        pickle.dump(plot_dict, handle)

def main():
    args = parse_arguments()
    currentdir = os.getcwd()
    model_folder_list = get_model_folders(currentdir, args.modellist, args.include_CONT)
    model_folder_list = filter_cont_models(model_folder_list, args.include_CONT)

    sample_df = pd.read_csv(os.path.join(currentdir, model_folder_list[0], '_INT_dict.csv'))
    lig_id_list = sorted(list(set(sample_df['lig_id'].values.tolist())))
    rct_id_list = sorted(list(set(sample_df['rct_id'].values.tolist())))

    output_dict = initialize_output_dict(lig_id_list, rct_id_list)
    aggregate_data(model_folder_list, lig_id_list, rct_id_list, currentdir, output_dict)

    saving_dir = os.path.join(currentdir, "batchcheck_INT")
    create_saving_directory(saving_dir)

    plot_dict = plot_ligands(output_dict, model_folder_list, lig_id_list, saving_dir, args.mode)
    plot_reactants_vs_ligand(7, rct_id_list, output_dict, saving_dir, args.mode, model_folder_list)
    plot_ligands_vs_reactant(1, lig_id_list, output_dict, saving_dir, args.mode, model_folder_list)

    save_plot_dict(plot_dict, saving_dir)

if __name__ == "__main__":
    main()
