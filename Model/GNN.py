import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from MPNN_layer import MPNN

class AdvancedModel(nn.Module):
    def __init__(self, config):
        super(AdvancedModel, self).__init__()

        self.gnn_layer = MPNN(config['MPNN'])

        self.layer_dict = nn.ModuleDict()

        for layer in ['dense_E1', 'dense_E2', 'dense_output']:
            layer_config = config[layer]

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

            self.layer_dict[layer] = nn.Sequential(
                dense_layer,
                activation_fn
            )

        self.h_constant = 6.62607015 * (10 ** -34)
        self.kb_constant = 1.380649 * (10 ** -23)
        self.j_to_kcal = 4184
        self.R_constant = 8.31
        self.temperature = 273 + 80
        self.s_to_h = 3.6

        self.use_arrhenius = True
        self.return_vertex_value = False
        self.analysis_mode = False

    def calculate_arrhenius(self, Ea):
        return self.kb_constant * self.temperature / self.h_constant / self.s_to_h * torch.exp(-Ea * self.j_to_kcal / self.R_constant / self.temperature)

    def enable_arrhenius(self):
        self.use_arrhenius = True

    def disable_arrhenius(self):
        self.use_arrhenius = False

    def enable_vertex_value(self):
        self.return_vertex_value = True

    def disable_vertex_value(self):
        self.return_vertex_value = False

    def forward(self, input_data, return_full_output=False, return_intermediate_output=False, device=None, training=None):
        intermediate_outputs = {}

        reactant_data, ligand_data = input_data

        # Process reactant data
        reactant_tensor = torch.tensor(reactant_data, dtype=torch.float32)

        # Process ligand data
        if device is not None:
            ligand_tensor = ligand_data.to(device)
        else:
            ligand_tensor = ligand_data

        ligand_tensor = self.gnn_layer(ligand_tensor)
        intermediate_outputs['ligand_inputs'] = ligand_tensor

        # Optionally return vertex value directly
        if self.return_vertex_value:
            return ligand_tensor

        if self.analysis_mode:
            return reactant_data, ligand_tensor

        # Dense layer E1
        E1_inputs = torch.cat([reactant_tensor, ligand_tensor], dim=1)
        E1_outputs = self.layer_dict['dense_E1'](E1_inputs)
        intermediate_outputs['inputs_E1'] = E1_outputs

        # Dense layer E2
        E2_outputs = self.layer_dict['dense_E2'](ligand_tensor)
        intermediate_outputs['inputs_E2'] = E2_outputs

        # Concatenate E1 and E2
        concatenated_inputs = torch.cat([E1_outputs, E2_outputs], dim=1)
        final_outputs = self.layer_dict['dense_output'](concatenated_inputs)

        if self.use_arrhenius:
            final_outputs = self.calculate_arrhenius(final_outputs)

        if return_full_output:
            return final_outputs, ligand_tensor, self.layer_dict
        if return_intermediate_output:
            return final_outputs, ligand_tensor, self.layer_dict, intermediate_outputs
        return final_outputs
