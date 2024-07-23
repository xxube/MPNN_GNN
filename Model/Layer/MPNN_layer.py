import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class MPNN(nn.Module):
    
    def __init__(self, _setup_dict):
        super(MPNN, self).__init__()

        message_dim = _setup_dict['message_dim']
        hidden_dim = _setup_dict['hidden_dim']
        layer_num = _setup_dict['layer_num']
        use_bias = _setup_dict['use_bias']
        activation = _setup_dict['activation']
        activation_alpha = _setup_dict.get('activation_alpha', 0.01)
        include_edge_feat = _setup_dict['include_edge_feat']
        repeat_msgANDhidden_layer  = _setup_dict['repeat_msgANDhidden_layer']
        
        self.layer_num = layer_num
        self.include_edge_feat = include_edge_feat
        self.repeat_msgANDhidden_layer = repeat_msgANDhidden_layer

        if activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU(activation_alpha)
        else:
            self.activation = getattr(F, activation)
        
        self.current_layer = 0
        self.dense_message_list = nn.ModuleList()
        self.dense_hidden_list = nn.ModuleList()

        if not repeat_msgANDhidden_layer:
            for _ in range(self.layer_num):
                self.dense_message_list.append(nn.Linear(message_dim, message_dim, bias=use_bias))
                self.dense_hidden_list.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
        else:
            self.dense_message_list.append(nn.Linear(message_dim, message_dim, bias=use_bias))
            self.dense_hidden_list.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))

    def message_func(self, edges):
        if self.include_edge_feat:
            m_input = torch.cat([edges.src['h_n'], edges.dst['h_n'], edges.data['h_e']], dim=-1)
        else:
            m_input = torch.cat([edges.src['h_n'], edges.dst['h_n']], dim=-1)
        m_out = self.activation(self.dense_message_list[self.current_layer](m_input))
        return {'m': m_out}

    def update_func(self, nodes):
        h_input = torch.cat([nodes.data['m_sum'], nodes.data['h_n']], dim=-1)
        h_out = self.activation(self.dense_hidden_list[self.current_layer](h_input))
        if not self.repeat_msgANDhidden_layer:
            self.current_layer += 1
        return {'h_n': h_out}
        
    def forward(self, graph_list):
        graph = graph_list

        with graph.local_scope():
            for _ in range(self.layer_num):
                graph.update_all(
                    self.message_func,
                    fn.sum('m', 'm_sum'),
                    self.update_func
                )
            
            nodes_label_list = torch.unique(graph.ndata['id'])
            nodes_label_len = nodes_label_list.shape[0]
            split_pattern = [nodes_label_len] * (graph.num_nodes() // nodes_label_len)

            self.current_layer = 0
            
            return torch.squeeze(
                torch.split(graph.ndata['h_n'], split_pattern)
            )


