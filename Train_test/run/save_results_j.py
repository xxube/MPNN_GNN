import json
import pandas as pd
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dirname', type=str)
    parser.add_argument('_optuna_filename', type=str)
    args = parser.parse_args()
    args._optuna_filename = args._optuna_filename.replace(".py", "")
    return args

def load_trial_setup(currentdir, model_dirname):
    with open(os.path.join(currentdir, model_dirname, model_dirname + "_setting"), "r") as input_file:
        return json.load(input_file)

def import_optuna_module(currentdir, _optuna_filename):
    import sys
    sys.path.insert(1, currentdir)
    return __import__(_optuna_filename)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def extract_model_architecture(model, train_features):
    model_arch = {}
    model.eval()
    with torch.no_grad():
        _, dense_dict, _ = model(train_features, return_layers_int=True)
        for layer_name, layer in dense_dict.items():
            W_b = [param.detach().numpy() for param in layer.parameters()]
            if len(W_b) == 2:
                _W, _b = W_b
            elif len(W_b) == 1:
                _W = W_b[0]
                _b = np.zeros(_W.shape[1:])
            if _W.shape == (1, 1):
                _W = float(_W[0])
            else:
                _W = _W.tolist()
            if isinstance(_b, np.ndarray) and _b.shape == (1,):
                _b = _b[0]
            model_arch[layer_name] = {'weights': _W, 'bias': _b}
    return model_arch

def process_data_and_save_results(model, _output_dict, weighting_rule, model_arch, model_dirname, currentdir):
    output_df = None
    for _data_type in ['train', 'vali', 'test']:
        _tag = _output_dict[f'{_data_type}_tag'].values.tolist()
        rct_id = [_x.split("_")[0] for _x in _tag]
        lig_id = [_x.split("_")[1] for _x in _tag]
        _features = torch.tensor(_output_dict[f'{_data_type}_features'], dtype=torch.float32)
        _values = _output_dict[f'{_data_type}_values']

        with torch.no_grad():
            y_pred, _, _INT_dict_pred = model(_features, return_layers_int=True)

            _outputs = {
                "data_type": [_data_type] * len(_tag),
                "tag": _tag,
                "rct_id": rct_id,
                "lig_id": lig_id,
                "y_pred": y_pred.squeeze().numpy(),
                "y_true": _values,
            }
            for k, v in _INT_dict_pred.items():
                _v = v.squeeze().numpy()
                if isinstance(_v[0], np.ndarray):
                    _v = [list(_x) for _x in _v]
                _outputs[k] = _v

                if k in weighting_rule.keys():
                    _layer, _idx = weighting_rule[k].split("|")
                    _idx = int(_idx)
                    weighting = model_arch[_layer]['weights'][_idx]
                    _outputs[k + "_Weighted"] = [weighting * _z for _z in _v]

            current_output_df = pd.DataFrame.from_dict(_outputs)

            if output_df is None:
                output_df = current_output_df
            else:
                output_df = output_df.append(current_output_df, ignore_index=True)

    output_df.sort_values(by=["rct_id", "lig_id"])
    output_df.to_csv(os.path.join(currentdir, model_dirname, "_INT_dict.csv"), index=False)

def save_model_architecture(model_arch, model_dirname, currentdir):
    _result = {"model_arch": model_arch}
    with open(os.path.join(currentdir, model_dirname, "_TrainedModel_INFO"), "w+") as output:
        json.dump(_result, output, indent=4, cls=NpEncoder)

def main():
    args = parse_args()
    model_dirname = args.model_dirname
    currentdir = os.path.dirname(os.path.realpath(__file__))

    trial_setup_dict = load_trial_setup(currentdir, model_dirname)
    _optuna = import_optuna_module(currentdir, args._optuna_filename)

    _output_dict = _optuna.get_data(
        trial_setup_dict=trial_setup_dict,
        batch_size=10000000,
        test_split=trial_setup_dict['test_split'],
        vali_split=trial_setup_dict['vali_split'],
        currentdir=currentdir,
        as_dataset=False,
    )

    train_features = torch.tensor(_output_dict['train_features'], dtype=torch.float32)
    model = _optuna.create_model(trial_setup_dict)
    optimizer = _optuna.create_optimizer(trial_setup_dict, model.parameters())

    load_checkpoint(model, optimizer, os.path.join(currentdir, model_dirname, model_dirname + '_checkpoint.pth'))
    model_arch = extract_model_architecture(model, train_features)

    process_data_and_sa
