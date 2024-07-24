import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNeuralNetwork(nn.Module):
    def __init__(self, config):
        super(CustomNeuralNetwork, self).__init__()
        self.layers = nn.ModuleDict()

        for block in ['dense_E1', 'dense_E2', 'dense_output']:
            block_config = config[block]

            units = block_config['units']
            activation_fn = block_config['activation']
            use_bias = block_config['use_bias']

            regularizer = block_config['kernel_regularizer']
            if regularizer == 'None':
                regularizer = None
            elif isinstance(regularizer, dict):
                for reg_type, reg_value in regularizer.items():
                    if reg_type == "l1":
                        regularizer = nn.L1Loss()
                    elif reg_type == "l2":
                        regularizer = nn.MSELoss()

            constraint = block_config['kernel_constraint']
            if constraint == 'NonNeg':
                constraint = lambda x: torch.clamp(x, min=0.0)
            elif constraint == 'None':
                constraint = None

            dense_layer = nn.Linear(units, units, bias=use_bias)
            activation = nn.LeakyReLU() if activation_fn.lower() == "leakyrelu" else getattr(F, activation_fn)

            self.layers[block] = nn.Sequential(
                dense_layer,
                activation
            )

        self.h_constant = 6.62607015 * (10 ** -34)
        self.kb_constant = 1.380649 * (10 ** -23)
        self.j_to_kcal = 4184
        self.R_constant = 8.31
        self.temperature = 273 + 80
        self.s_to_h = 3.6

        self.enable_arrhenius()
        self.disable_vertex_value()
        self.for_analysis = False

    def arrhenius_formula(self, Ea):
        return self.kb_constant * self.temperature / self.h_constant / self.s_to_h * torch.exp(-Ea * self.j_to_kcal / self.R_constant / self.temperature)

    def enable_arrhenius(self):
        self.use_arrhenius = True

    def disable_arrhenius(self):
        self.use_arrhenius = False

    def enable_vertex_value(self):
        self.use_vertex_value = True

    def disable_vertex_value(self):
        self.use_vertex_value = False

    def forward(self, input_data, return_full=False, return_layers_info=False, device=None, training=None):
        intermediates = {}

        reactant_data, ligand_data = input_data
        
        # Process reactant data
        reactant_inputs = torch.tensor(reactant_data, dtype=torch.float32)

        # Process ligand data
        ligand_inputs = torch.tensor(ligand_data, dtype=torch.float32)
        intermediates['ligand_inputs'] = ligand_inputs
        
        # Optionally return vertex value directly
        if self.use_vertex_value:
            return ligand_inputs
        if self.for_analysis:
            return reactant_data, ligand_inputs

        # Dense layer E1
        inputs_E1 = torch.cat([reactant_inputs, ligand_inputs], dim=1)
        inputs_E1 = self.layers['dense_E1'](inputs_E1)
        intermediates['inputs_E1'] = inputs_E1

        # Dense layer E2
        inputs_E2 = self.layers['dense_E2'](ligand_inputs)
        intermediates['inputs_E2'] = inputs_E2

        # Concatenate E1 and E2
        concatenated_inputs = torch.cat([inputs_E1, inputs_E2], dim=1)
        outputs = self.layers['dense_output'](concatenated_inputs)

        if self.use_arrhenius:
            outputs = self.arrhenius_formula(outputs)

        if return_full:
            return outputs, ligand_inputs, self.layers
        if return_layers_info:
            return outputs, ligand_inputs, self.layers, intermediates
        return outputs
