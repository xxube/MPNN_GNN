import torch
import torch.nn as nn
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, _setup_dict):
        super(MODEL, self).__init__()

        # self.MPNN1 = MPNN(_setup_dict['MPNN'])  # Ignored as per the comment in the code

        self.dense_dict = nn.ModuleDict()

        for _block in ['dense_E1', 'dense_E2', 'dense_output']:
            _block_dict = _setup_dict[_block]

            units = _block_dict['units']
            activation = _block_dict['activation']
            use_bias = _block_dict['use_bias']

            kernel_regularizer = _block_dict['kernel_regularizer']
            if kernel_regularizer == 'None':
                kernel_regularizer = None
            elif isinstance(kernel_regularizer, dict):
                for k, v in kernel_regularizer.items():
                    _type = k
                    _value = v
                if _type == "l1":
                    kernel_regularizer = nn.L1Loss()
                elif _type == "l2":
                    kernel_regularizer = nn.MSELoss()

            kernel_constraint = _block_dict['kernel_constraint']
            if kernel_constraint == 'NonNeg':
                kernel_constraint = lambda x: torch.clamp(x, min=0.0)
            elif kernel_constraint == 'None':
                kernel_constraint = None

            dense_layer = nn.Linear(units, units, bias=use_bias)
            if activation.lower() == "leakyrelu":
                activation_fn = nn.LeakyReLU()
            else:
                activation_fn = getattr(F, activation)

            self.dense_dict[_block] = nn.Sequential(
                dense_layer,
                activation_fn
            )
        
        self._h = 6.62607015 * (10 ** -34)
        self._kb = 1.380649 * (10 ** -23)
        self._J_to_kcal = 4184
        self._R = 8.31
        self._T = 273 + 80
        self._1000s_to_h = 3.6

        self.useArrheniusEq = True
        self.useVertexValue = False
        self.forAnalyze = False

    def ArrheniusEq(self, Ea):
        return self._kb * self._T / self._h / self._1000s_to_h * torch.exp(-Ea * self._J_to_kcal / self._R / self._T)

    def stopuseArrheniusEq(self):
        self.useArrheniusEq = False

    def startuseArrheniusEq(self):
        self.useArrheniusEq = True

    def stopuseVertexValue(self):
        self.useVertexValue = False

    def startuseVertexValue(self):
        self.useVertexValue = True

    def forward(self, input_data, return_FCNN=False, return_LayersInt=False, _device=None, training=None):
        _INT_dict = {}

        rct_inputs_data, lig_inputs_data = input_data
        
        # treat rct
        rct_inputs = torch.tensor(rct_inputs_data, dtype=torch.float32)

        # treat lig (OLD)
        # if _device is not None:
        #     lig_inputs = lig_inputs_data.to(_device)
        # else:
        #     lig_inputs = lig_inputs_data
        # lig_inputs = self.MPNN1(lig_inputs)

        # treat lig
        lig_inputs = torch.tensor(lig_inputs_data, dtype=torch.float32)
        _INT_dict['lig_inputs'] = lig_inputs
        # (optional) return vertex value directly
        if self.useVertexValue:
            return lig_inputs
        if self.forAnalyze:
            return rct_inputs_data, lig_inputs

        # E1
        inputs_E1 = torch.cat([rct_inputs, lig_inputs], dim=1)
        inputs_E1 = self.dense_dict['dense_E1'](inputs_E1)
        _INT_dict['inputs_E1'] = inputs_E1

        # E2
        inputs_E2 = self.dense_dict['dense_E2'](lig_inputs)
        _INT_dict['inputs_E2'] = inputs_E2

        # concat E1 and E2
        inputs_E1_E2 = torch.cat([inputs_E1, inputs_E2], dim=1)  # torch.cat([inputs_E1, -inputs_E2], dim=1)
        outputs = self.dense_dict['dense_output'](inputs_E1_E2)

        if self.useArrheniusEq:
            outputs = self.ArrheniusEq(outputs)

        if return_FCNN:
            return outputs, lig_inputs, self.dense_dict
        if return_LayersInt:
            return outputs, lig_inputs, self.dense_dict, _INT_dict
        return outputs
