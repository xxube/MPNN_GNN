import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MPNN(nn.Module):
    
    def __init__(self, config):
        super(MPNN, self).__init__()

        self.message_dim = config['message_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['layer_num']
        self.use_bias = config['use_bias']
        self.include_edge_feat = config['include_edge_feat']
        self.repeat_layers = config['repeat_msgANDhidden_layer']

        activation = config['activation']
        self.activation_alpha = config.get('activation_alpha', 0.01)
        self.activation = nn.LeakyReLU(self.activation_alpha) if activation.lower() == "leakyrelu" else getattr(F, activation)
        
        self.current_layer_idx = 0
        self.message_layers = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        if self.repeat_layers:
            self.message_layers.append(nn.Linear(self.message_dim, self.message_dim, bias=self.use_bias))
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))
        else:
            for _ in range(self.num_layers):
                self.message_layers.append(nn.Linear(self.message_dim, self.message_dim, bias=self.use_bias))
                self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))

    def compute_message(self, edges):
        if self.include_edge_feat:
            message_input = torch.cat([edges.src['h_n'], edges.dst['h_n'], edges.data['h_e']], dim=-1)
        else:
            message_input = torch.cat([edges.src['h_n'], edges.dst['h_n']], dim=-1)
        message_output = self.activation(self.message_layers[self.current_layer_idx](message_input))
        return {'message': message_output}

    def apply_update(self, nodes):
        hidden_input = torch.cat([nodes.data['message_sum'], nodes.data['h_n']], dim=-1)
        hidden_output = self.activation(self.hidden_layers[self.current_layer_idx](hidden_input))
        if not self.repeat_layers:
            self.current_layer_idx += 1
        return {'h_n': hidden_output}
        
    def forward(self, graph):
        with graph.local_scope():
            for _ in range(self.num_layers):
                graph.update_all(
                    self.compute_message,
                    fn.sum('message', 'message_sum'),
                    self.apply_update
                )
            
            unique_node_labels = torch.unique(graph.ndata['id'])
            num_unique_labels = unique_node_labels.shape[0]
            split_sizes = [num_unique_labels] * (graph.num_nodes() // num_unique_labels)

            self.current_layer_idx = 0
            
            return torch.squeeze(
                torch.split(graph.ndata['h_n'], split_sizes)
            )

