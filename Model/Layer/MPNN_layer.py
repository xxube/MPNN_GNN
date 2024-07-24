import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MPNN(nn.Module):

    def __init__(self, config):
        super(MPNN, self).__init__()

        self.msg_dim = config['message_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['layer_num']
        self.use_bias = config['use_bias']
        self.include_edge_features = config['include_edge_feat']
        self.repeat_layers = config['repeat_msgANDhidden_layer']

        self.activation_type = config['activation']
        self.activation_alpha = config.get('activation_alpha', 0.01)
        self.activation = nn.LeakyReLU(self.activation_alpha) if self.activation_type.lower() == "leakyrelu" else getattr(F, self.activation_type)
        
        self.current_layer = 0
        self.msg_layers = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        if self.repeat_layers:
            self.msg_layers.append(nn.Linear(self.msg_dim, self.msg_dim, bias=self.use_bias))
            self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))
        else:
            for _ in range(self.num_layers):
                self.msg_layers.append(nn.Linear(self.msg_dim, self.msg_dim, bias=self.use_bias))
                self.hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))

    def compute_messages(self, edges):
        if self.include_edge_features:
            msg_input = torch.cat([edges.src['h_n'], edges.dst['h_n'], edges.data['h_e']], dim=-1)
        else:
            msg_input = torch.cat([edges.src['h_n'], edges.dst['h_n']], dim=-1)
        msg_output = self.activation(self.msg_layers[self.current_layer](msg_input))
        return {'message': msg_output}

    def update_nodes(self, nodes):
        hidden_input = torch.cat([nodes.data['message_sum'], nodes.data['h_n']], dim=-1)
        hidden_output = self.activation(self.hidden_layers[self.current_layer](hidden_input))
        if not self.repeat_layers:
            self.current_layer += 1
        return {'h_n': hidden_output}
        
    def forward(self, graph):
        with graph.local_scope():
            for _ in range(self.num_layers):
                graph.update_all(
                    self.compute_messages,
                    fn.sum('message', 'message_sum'),
                    self.update_nodes
                )
            
            unique_node_ids = torch.unique(graph.ndata['id'])
            num_unique_nodes = unique_node_ids.shape[0]
            split_sizes = [num_unique_nodes] * (graph.num_nodes() // num_unique_nodes)

            self.current_layer = 0
            
            return torch.squeeze(
                torch.split(graph.ndata['h_n'], split_sizes)
            )
