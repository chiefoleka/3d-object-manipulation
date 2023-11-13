import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class BasicModelNorm(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.linear = torch.nn.Linear(num_classes, 3)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index = data.x_norms, data.edge_index

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = scatter_mean(x, torch.tensor([0]), dim=0)

        return self.linear(x)

class BasicModelNormLarge(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 1024)
        self.conv2 = GCNConv(1024, 16)
        self.conv3 = GCNConv(16, num_classes)
        self.linear = torch.nn.Linear(num_classes, 3)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index = data.x_norms, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)

        x = scatter_mean(x, torch.tensor([0]), dim=0)

        return self.linear(x)

class BasicModelNormExtraLarge(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 4096)
        self.conv2 = GCNConv(4096, 1024)
        self.conv3 = GCNConv(1024, 256)
        self.conv4 = GCNConv(256, 64)
        self.conv5 = GCNConv(64, num_classes)
        self.linear = torch.nn.Linear(num_classes, 3)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index = data.x_norms, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)

        x = scatter_mean(x, torch.tensor([0]), dim=0)

        return self.linear(x)
