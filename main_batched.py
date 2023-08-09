import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset_loader import DatasetLoader
from loss_plotter import plot_loss
from config import config as get_config
# from model import GCN
from model2 import GCN

loader = DatasetLoader()
test_dataset = loader.load(get_config('test_path'))
train_dataset = loader.load(get_config('train_path'))

test_data = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
for epoch in range(20):
    total_loss = 0
    gcn_model.train()

    # Figure out the batch size and loop through a batch here and not all data
    counterTrain = 1
    counterTest = 1
    for data in train_data:
        optimizer.zero_grad()
        out = gcn_model(data)

        loss = F.l1_loss(out[0], data.y)
        total_loss = total_loss + loss
        loss.backward()
        optimizer.step()

        counterTrain += 1

    eval_loss = 0
    gcn_model.eval()
    for data in test_data:
        pred = gcn_model(data)
        loss = F.l1_loss(pred[0], data.y)
        eval_loss = eval_loss + loss

        counterTest += 1

    # save mean loss here
    mean_loss.append(total_loss / counterTrain)
    mean_eval_loss.append(eval_loss / counterTest)

# plot_loss(mean_loss, 'Training Loss')
# plot_loss(mean_eval_loss, 'Evaluation Loss')
print('Final training loss', mean_loss[-1])
print('Final evaluation loss', mean_eval_loss[-1])

# Get all the test data and get the mean value of the loss
# Get the mean value on the loss from the test and see the final loss
# Save the loss over time and plot it to see if the model is improving
# plot the training
# calculate final mean loss
# think about preprocessing and adjustment of hyperparameters to improve
gcn_model.eval()

# eval_loss = []
# for data in test_data.dataset:
#     pred = gcn_model(data)
#     loss = F.l1_loss(pred[0], data.y)
#     eval_loss.append(loss)

# plot_loss(mean_loss)
# plot_loss(eval_loss, 'Evaluation Loss')
