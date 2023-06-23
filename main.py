import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset_loader import DatasetLoader
from config import config
from model import GCN

loader = DatasetLoader()
test_dataset = loader.load(config('test_path'))
train_dataset = loader.load(config('train_path'))

test_data = DataLoader(test_dataset, batch_size=4, shuffle=True)
train_data = DataLoader(train_dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn_model = GCN().to(device)
data = train_data.dataset[0].to(device)
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

# this is incomplete.
# need to figureout how to setup the graph to fit into this model
# alternatively, need to figure out how to calculate the loss and accuracy
gcn_model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = gcn_model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

gcn_model.eval()
pred = gcn_model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
