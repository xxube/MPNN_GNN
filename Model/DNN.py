import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()

        # self.GNN1 = MPNN(config['MPNN'])  # Ignored as per the comment in the code

        self.layers_dict = nn.ModuleDict()

        for block in ['dense_E1', 'dense_E2', 'dense_output']:
            block_config = config[block]

            units = block_config['units']
            activation = block_config['activation']
            use_bias = block_config['use_bias']

            kernel_regularizer = block_config['kernel_regularizer']
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

            kernel_constraint = block_config['kernel_constraint']
            if kernel_constraint == 'NonNeg':
                kernel_constraint = lambda x: torch.clamp(x, min=0.0)
            elif kernel_constraint == 'None':
                kernel_constraint = None

            dense_layer = nn.Linear(units, units, bias=use_bias)
            if activation.lower() == "leakyrelu":
                activation_fn = nn.LeakyReLU()
            else:
                activation_fn = getattr(F, activation)

            self.layers_dict[block] = nn.Sequential(
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
        self.use_vertex_value = False
        self.for_analysis = False

    def arrhenius_equation(self, Ea):
        return self.kb_constant * self.temperature / self.h_constant / self.s_to_h * torch.exp(-Ea * self.j_to_kcal / self.R_constant / self.temperature)

    def disable_arrhenius(self):
        self.use_arrhenius = False

    def enable_arrhenius(self):
        self.use_arrhenius = True

    def disable_vertex_value(self):
        self.use_vertex_value = False

    def enable_vertex_value(self):
        self.use_vertex_value = True

    def forward(self, input_data, return_full=False, return_layers_info=False, device=None, training=None):
        intermediate_dict = {}

        reactant_data, ligand_data = input_data
        
        # process reactant data
        reactant_inputs = torch.tensor(reactant_data, dtype=torch.float32)

        # process ligand data (OLD)
        # if device is not None:
        #     ligand_inputs = ligand_data.to(device)
        # else:
        #     ligand_inputs = ligand_data
        # ligand_inputs = self.GNN1(lig_inputs)

        # process ligand data
        ligand_inputs = torch.tensor(ligand_data, dtype=torch.float32)
        intermediate_dict['ligand_inputs'] = ligand_inputs
        
        # optionally return vertex value directly
        if self.use_vertex_value:
            return ligand_inputs
        if self.for_analysis:
            return reactant_data, ligand_inputs

        # Dense layer E1
        inputs_E1 = torch.cat([reactant_inputs, ligand_inputs], dim=1)
        inputs_E1 = self.layers_dict['dense_E1'](inputs_E1)
        intermediate_dict['inputs_E1'] = inputs_E1

        # Dense layer E2
        inputs_E2 = self.layers_dict['dense_E2'](ligand_inputs)
        intermediate_dict['inputs_E2'] = inputs_E2

        # concatenate E1 and E2
        concatenated_inputs = torch.cat([inputs_E1, inputs_E2], dim=1)
        outputs = self.layers_dict['dense_output'](concatenated_inputs)

        if self.use_arrhenius:
            outputs = self.arrhenius_equation(outputs)

        if return_full:
            return outputs, ligand_inputs, self.layers_dict
        if return_layers_info:
            return outputs, ligand_inputs, self.layers_dict, intermediate_dict
        return outputs
