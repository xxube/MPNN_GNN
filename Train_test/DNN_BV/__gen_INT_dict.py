import os
import json
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dirname', type=str)
    parser.add_argument('_optuna_filename', type=str)
    args = parser.parse_args()
    args._optuna_filename = args._optuna_filename.replace(".py", "")
    return args

def load_optuna_module(currentdir, _optuna_filename):
    sys.path.insert(1, currentdir)
    return __import__(_optuna_filename)

def restore_checkpoint(model, optimizer, model_dir):
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def extract_model_architecture(model):
    model_arch = {}
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            model_arch[name] = param.detach().numpy().tolist()
    return model_arch

def process_data(model, features, labels, weighting_rule, model_arch):
    model.eval()
    with torch.no_grad():
        y_pred = model(features).squeeze().numpy()
    outputs = {
        "y_pred": y_pred,
        "y_true": labels
    }
    for name, value in weighting_rule.items():
        layer, idx = value.split("|")
        idx = int(idx)
        weights = model_arch[f"{layer}.weight"][idx]
        outputs[f"{name}_Weighted"] = [weights * x for x in y_pred]
    return outputs

def main():
    args = parse_arguments()
    currentdir = os.getcwd()

    with open(os.path.join(currentdir, args.model_dirname, f"{args.model_dirname}_setting"), "r") as input_file:
        trial_setup_dict = json.load(input_file)

    _optuna = load_optuna_module(currentdir, args._optuna_filename)

    model_dir = os.path.join(currentdir, args.model_dirname, args.model_dirname)
    data_dict = _optuna.get_data(
        trial_setup_dict=trial_setup_dict,
        batch_size=10000000,
        test_split=trial_setup_dict['test_split'],
        vali_split=trial_setup_dict['vali_split'],
        currentdir=currentdir,
        as_dataset=False,
    )

    model = _optuna.create_model(trial_setup_dict)
    optimizer = _optuna.create_optimizer(trial_setup_dict)
    model, optimizer = restore_checkpoint(model, optimizer, os.path.join(model_dir, "model.pth"))

    model_arch = extract_model_architecture(model)

    output_df = None
    for data_type in ['train', 'vali', 'test']:
        tags = data_dict[f'{data_type}_tag'].values.tolist()
        rct_id = [x.split("_")[0] for x in tags]
        lig_id = [x.split("_")[1] for x in tags]
        features = torch.tensor(data_dict[f'{data_type}_features']).float()
        values = torch.tensor(data_dict[f'{data_type}_values']).float()

        outputs = process_data(model, features, values, weighting_rule, model_arch)
        outputs.update({
            "data_type": [data_type] * len(tags),
            "tag": tags,
            "rct_id": rct_id,
            "lig_id": lig_id,
        })

        current_output_df = pd.DataFrame.from_dict(outputs)
        if output_df is None:
            output_df = current_output_df
        else:
            output_df = pd.concat([output_df, current_output_df], ignore_index=True)

    output_df.sort_values(by=["rct_id", "lig_id"], inplace=True)
    output_df.to_csv(os.path.join(currentdir, args.model_dirname, "_INT_dict.csv"), index=False)

    result = {"model_arch": model_arch}
    with open(os.path.join(currentdir, args.model_dirname, "_TrainedModel_INFO"), "w") as output_file:
        json.dump(result, output_file, indent=4)

if __name__ == "__main__":
    main()
