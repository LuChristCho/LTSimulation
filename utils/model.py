import torch
import torch.nn as nn
import torch_geometric as pyg
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        if layernorm:
            self.layers.append(nn.LayerNorm(input_size))
        
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden_size, output_size))
        
        # Init weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in 'Learning to Simulate Complex Physics'."""
    def __init__(self, hidden_size, layers):
        super().__init__(aggr='add') 
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature):
        # x: [N, hidden_size], edge_index: [2, E], edge_feature: [E, hidden_size]
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x, x

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        edge_out, aggr = inputs
        return edge_out, super().aggregate(aggr, index, ptr, dim_size)

class LearnedSimulator(nn.Module):
    def __init__(self, hidden_size=128, n_mp_layers=10, num_particle_types=9, particle_type_dim=16, dim=2, window_size=6):
        super().__init__()
        self.window_size = window_size
        self.embed_type = nn.Embedding(num_particle_types, particle_type_dim)
        
        # Input: velocity seq (window-1 * dim) + boundary distance (4) + particle type embedding
        node_input_size = (window_size) * dim + 4 + particle_type_dim
        edge_input_size = 3  # Displacement (2) + distance (1)

        self.node_encoder = MLP(node_input_size, hidden_size, hidden_size, 3)
        self.edge_encoder = MLP(edge_input_size, hidden_size, hidden_size, 3)
        
        self.layers = nn.ModuleList([InteractionNetwork(hidden_size, 3) for _ in range(n_mp_layers)])
        
        self.decoder = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)

    def forward(self, data):
        # Embed particle type
        node_type_emb = self.embed_type(data.x)
        
        # Concatenate inputs for nodes
        node_feat = torch.cat((data.pos, node_type_emb), dim=-1)
        
        # Encode
        node_emb = self.node_encoder(node_feat)
        edge_emb = self.edge_encoder(data.edge_attr)
        
        # Process (Message Passing)
        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, data.edge_index, edge_emb)
            
        # Decode
        pred_acc = self.decoder(node_emb)
        return pred_acc