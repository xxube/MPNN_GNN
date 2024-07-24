import os
import json
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('optuna_file', type=str)
    args = parser.parse_args()
    args.optuna_file = args.optuna_file.replace(".py", "")
    return args

def import_optuna_module(current_directory, optuna_file):
    sys.path.insert(1, current_directory)
    return __import__(optuna_file)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def get_model_architecture(model):
    architecture = {}
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            architecture[name] = param.detach().numpy().tolist()
    return architecture

def evaluate_model(model, features, labels, weighting_rule, architecture):
    model.eval()
    with torch.no_grad():
        y_pred = model(features).squeeze().numpy()
    results = {
        "y_pred": y_pred,
        "y_true": labels
    }
    for name, rule in weighting_rule.items():
        layer, idx = rule.split("|")
        idx = int(idx)
        weights = architecture[f"{layer}.weight"][idx]
        results[f"{name}_Weighted"] = [weights * x for x in y_pred]
    return results

def main():
    args = get_args()
    current_directory = os.getcwd()

    with open(os.path.join(current_directory, args.model_directory, f"{args.model_directory}_setting"), "r") as input_file:
        trial_setup = json.load(input_file)

    optuna_module = import_optuna_module(current_directory, args.optuna_file)

    model_path = os.path.join(current_directory, args.model_directory, args.model_directory)
    data = optuna_module.get_data(
        trial_setup_dict=trial_setup,
        batch_size=10000000,
        test_split=trial_setup['test_split'],
        vali_split=trial_setup['vali_split'],
        currentdir=current_directory,
        as_dataset=False,
    )

    model = optuna_module.create_model(trial_setup)
    optimizer = optuna_module.create_optimizer(trial_setup)
    model, optimizer = load_checkpoint(model, optimizer, os.path.join(model_path, "model.pth"))

    architecture = get_model_architecture(model)

    output_dataframe = None
    for data_type in ['train', 'vali', 'test']:
        tags = data[f'{data_type}_tag'].values.tolist()
        rct_ids = [tag.split("_")[0] for tag in tags]
        lig_ids = [tag.split("_")[1] for tag in tags]
        features = torch.tensor(data[f'{data_type}_features']).float()
        labels = torch.tensor(data[f'{data_type}_values']).float()

        results = evaluate_model(model, features, labels, weighting_rule, architecture)
        results.update({
            "data_type": [data_type] * len(tags),
            "tag": tags,
            "rct_id": rct_ids,
            "lig_id": lig_ids,
        })

        current_output_df = pd.DataFrame.from_dict(results)
        if output_dataframe is None:
            output_dataframe = current_output_df
        else:
            output_dataframe = pd.concat([output_dataframe, current_output_df], ignore_index=True)

    output_dataframe.sort_values(by=["rct_id", "lig_id"], inplace=True)
    output_dataframe.to_csv(os.path.join(current_directory, args.model_directory, "_INT_dict.csv"), index=False)

    model_info = {"model_architecture": architecture}
    with open(os.path.join(current_directory, args.model_directory, "_TrainedModel_INFO"), "w") as output_file:
        json.dump(model_info, output_file, indent=4)

if __name__ == "__main__":
    main()
