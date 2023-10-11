import time
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from loader.dataset import Dataset
from utils.config import config as get_config
from models.basic import BasicModel as GCN
from utils.plotter import NumpyEncoder


loader = Dataset()
test_dataset = loader.load(get_config("test_path"))
train_dataset = loader.load(get_config("train_path"))

test_data = DataLoader(test_dataset, batch_size=2, shuffle=True)
train_data = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gcn_model = GCN(3, 1).to(device)

optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
gcn_model.train()
for data in train_data.dataset:
    data = data.to(device)

for data in test_data.dataset:
    data = data.to(device)

mean_loss = []
mean_eval_loss = []
itemized_train_loss = {}
itemized_test_loss = {}

for epoch in range(int(get_config('epochs', 25))):
    total_loss = 0
    gcn_model.train()
    # Figure out the batch size and loop through a batch here and not all data
    for data in train_data.dataset:
        optimizer.zero_grad()
        out = gcn_model(data)
        # print(f'out[0].shape: {out[0].shape}')
        # print(f'y.shape: {data.y.shape}')
        loss = F.l1_loss(out[0], data.y)
        total_loss = total_loss + loss
        loss.backward()
        optimizer.step()

        if data.filename not in itemized_train_loss:
            itemized_train_loss[data.filename] = []

        y_np = data.y.detach().numpy()
        o_np = out[0].detach().numpy()
        itemized_train_loss[data.filename].append(
            {
                "epoch": epoch,
                "agg_loss": loss.detach().numpy(),
                "loss": [y_np[0] - o_np[0], y_np[1] - o_np[1], y_np[2] - o_np[2]],
                "output": o_np,
                "expected": y_np,
            }
        )

    eval_loss = 0
    gcn_model.eval()
    for data in test_data.dataset:
        pred = gcn_model(data)
        loss = F.l1_loss(pred[0], data.y)
        eval_loss = eval_loss + loss

        if data.filename not in itemized_test_loss:
            itemized_test_loss[data.filename] = []

        y_np = data.y.detach().numpy()
        o_np = pred[0].detach().numpy()
        itemized_test_loss[data.filename].append(
            {
                "epoch": epoch,
                "agg_loss": loss.detach().numpy(),
                "predicted": o_np,
                "loss": [y_np[0] - o_np[0], y_np[1] - o_np[1], y_np[2] - o_np[2]],
                "expected": y_np,
            }
        )

    # save mean loss here
    mean_loss.append(total_loss / len(train_data.dataset))
    mean_eval_loss.append(eval_loss / len(test_data.dataset))

# plot_loss(mean_loss, 'Training Loss')
# plot_loss(mean_eval_loss, 'Evaluation Loss')
print("Final training loss", mean_loss[-1])
print("Final evaluation loss", mean_eval_loss[-1])

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f"outputs/test_loss{timestr}.json", "w") as f:
    json.dump(itemized_test_loss, f, cls=NumpyEncoder)

with open(f"outputs/train_loss{timestr}.json", "w") as f:
    json.dump(itemized_train_loss, f, cls=NumpyEncoder)
