import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from model import GCN

dataset = Planetoid(root='/tmp/Cora', name='Cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn_model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

gcn_model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = gcn_model(data)
    # y is the label of the defined dataset
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

gcn_model.eval()
pred = gcn_model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
