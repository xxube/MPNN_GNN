import json
import pandas as pd
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument('model_dirname', type=str)
parser.add_argument('_optuna_filename', type=str)

weighting_rule = {
    "inputs_E1": "dense_output|0",
    "inputs_E2": "dense_output|1",
}

args = parser.parse_args()
model_dirname = args.model_dirname
_optuna_filename = args._optuna_filename.replace(".py", "")

currentdir = os.path.dirname(os.path.realpath(__file__))

# Load trial setup dictionary
with open(os.path.join(currentdir, model_dirname, model_dirname + "_setting"), "r") as input_file:
    trial_setup_dict = json.load(input_file)

# Import the _optuna file as a module
import sys
_optuna_filedir, _optuna_filename = os.path.split(_optuna_filename)
sys.path.insert(1, currentdir)
_optuna = __import__(_optuna_filename)

# Get train/test data
_output_dict = _optuna.get_data(
    trial_setup_dict=trial_setup_dict,
    batch_size=10000000,
    test_split=trial_setup_dict['test_split'],
    vali_split=trial_setup_dict['vali_split'],
    currentdir=currentdir,
    as_dataset=False,
)

train_tag = _output_dict['train_tag']
train_features = torch.tensor(_output_dict['train_features'], dtype=torch.float32)
train_values = torch.tensor(_output_dict['train_values'], dtype=torch.float32)

# Define the model architecture and optimizer
model = _optuna.create_model(trial_setup_dict)
optimizer = _optuna.create_optimizer(trial_setup_dict, model.parameters())

# Load model checkpoint
checkpoint = torch.load(os.path.join(currentdir, model_dirname, model_dirname + '_checkpoint.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

print(f"Restored from {model_dirname + '_checkpoint.pth'}")

# Generate model architecture dictionary
model_arch = {}
model.eval()
with torch.no_grad():
    y_pred, dense_dict, _INT_dict_pred = model(train_features, return_layers_int=True)

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

# Process train/vali/test data and save results
output_df = None
for _data_type in ['train', 'vali', 'test']:
    _tag = _output_dict[f'{_data_type}_tag'].values.tolist()
    rct_id = [_x.split("_")[0] for _x in _tag]
    lig_id = [_x.split("_")[1] for _x in _tag]
    _features = torch.tensor(_output_dict[f'{_data_type}_features'], dtype=torch.float32)
    _values = _output_dict[f'{_data_type}_values']

    with torch.no_grad():
        y_pred, dense_dict, _INT_dict_pred = model(_features, return_layers_int=True)

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

_result = {
    "model_arch": model_arch,
}

output_df.sort_values(by=["rct_id", "lig_id"])
output_df.to_csv(os.path.join(currentdir, model_dirname, "_INT_dict.csv"), index=False)

# Save model architecture
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open(os.path.join(currentdir, model_dirname, "_TrainedModel_INFO"), "w+") as output:
    output.write(json.dumps(_result, indent=4, cls=NpEncoder))
