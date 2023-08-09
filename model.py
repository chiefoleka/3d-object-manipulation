import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.linear = torch.nn.Linear(num_classes, 3)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # print(f'self.conv1: {x.shape}')
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # print(f'self.conv2: {x.shape}')

        x = scatter_mean(x, torch.tensor([0]), dim=0)
        # print(f'scatter: {x.shape}')
        return self.linear(x)