import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from MPNN_layer import MPNN

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()

        self.gnn_layer = MPNN(config['MPNN'])

        self.dense_layers = nn.ModuleDict()

        for layer_name in ['dense_E1', 'dense_E2', 'dense_output']:
            layer_config = config[layer_name]

            units = layer_config['units']
            activation = layer_config['activation']
            use_bias = layer_config['use_bias']

            kernel_regularizer = layer_config['kernel_regularizer']
            if kernel_regularizer == 'None':
                kernel_regularizer = None
            elif isinstance(kernel_regularizer, dict):
                for key, value in kernel_regularizer.items():
                    reg_type = key
                    reg_value = value
                if reg_type == "l1":
                    kernel_regularizer = nn.L1Loss()
                elif reg_type == "l2":
                    kernel_regularizer = nn.MSELoss()

            kernel_constraint = layer_config['kernel_constraint']
            if kernel_constraint == 'NonNeg':
                kernel_constraint = lambda x: torch.clamp(x, min=0.0)
            elif kernel_constraint == 'None':
                kernel_constraint = None

            dense_layer = nn.Linear(units, units, bias=use_bias)

            if activation.lower() == "leakyrelu":
                activation_fn = nn.LeakyReLU()
            else:
                activation_fn = getattr(F, activation)

            self.dense_layers[layer_name] = nn.Sequential(
                dense_layer,
                activation_fn
            )
        
        self.h_constant = 6.62607015 * (10 ** -34)
        self.kb_constant = 1.380649 * (10 ** -23)
        self.j_to_kcal = 4184
        self.R_constant = 8.31
        self.temperature = 273 + 80
        self.s_to_h = 3.6

        self.use_arrhenius_eq = True
        self.use_vertex_value = False
        self.for_analysis = False

    def compute_arrhenius(self, Ea):
        return self.kb_constant * self.temperature / self.h_constant / self.s_to_h * torch.exp(-Ea * self.j_to_kcal / self.R_constant / self.temperature)

    def disable_arrhenius(self):
        self.use_arrhenius_eq = False

    def enable_arrhenius(self):
        self.use_arrhenius_eq = True

    def disable_vertex_value(self):
        self.use_vertex_value = False

    def enable_vertex_value(self):
        self.use_vertex_value = True

    def forward(self, input_data, return_full=False, return_intermediate=False, device=None, training=None):
        intermediate_dict = {}

        reactant_data, ligand_data = input_data

        # process reactant data
        reactant_inputs = torch.tensor(reactant_data, dtype=torch.float32)

        # process ligand data
        if device is not None:
            ligand_inputs = ligand_data.to(device)
        else:
            ligand_inputs = ligand_data

        ligand_inputs = self.gnn_layer(lig_inputs)
        intermediate_dict['ligand_inputs'] = lig_inputs

        # optionally return vertex value directly
        if self.use_vertex_value:
            return lig_inputs

        if self.for_analysis:
            return reactant_data, lig_inputs

        # Dense layer E1
        inputs_E1 = torch.cat([reactant_inputs, lig_inputs], dim=1)
        inputs_E1 = self.dense_layers['dense_E1'](inputs_E1)
        intermediate_dict['inputs_E1'] = inputs_E1

        # Dense layer E2
        inputs_E2 = self.dense_layers['dense_E2'](lig_inputs)
        intermediate_dict['inputs_E2'] = inputs_E2

        # concatenate E1 and E2
        concatenated_inputs = torch.cat([inputs_E1, inputs_E2], dim=1)
        outputs = self.dense_layers['dense_output'](concatenated_inputs)

        if self.use_arrhenius_eq:
            outputs = self.compute_arrhenius(outputs)

        if return_full:
            return outputs, lig_inputs, self.dense_layers
        if return_intermediate:
            return outputs, lig_inputs, self.dense_layers, intermediate_dict
        return outputs
