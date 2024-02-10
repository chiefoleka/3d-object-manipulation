import os
import time
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from loader.dataset import Dataset
from models.factories.model_factories import ModelFactory
from utils.config import config as get_config
from utils.plotter import NumpyEncoder, plot_loss


class Trainer:
    mean_loss = []
    mean_eval_loss = []
    itemized_train_loss = {}
    itemized_test_loss = {}

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_node_features = 3
        num_classes = 1
        self.model = ModelFactory.getModel(
            get_config("model_type"),
            num_node_features,
            num_classes,
            device
        )

        # Learning rate and weight decay
        learning_rate = float(get_config("learning_rate"))
        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=learning_rate, weight_decay=5e-4)
        self.load_data(device)

    def load_data(self, device):
        # load the data, converting STL binary to Graph
        loader = Dataset()
        self.test_dataset = loader.load(get_config("test_path"))
        self.train_dataset = loader.load(get_config("train_path"))
        self.train_data = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True)
        self.test_data = DataLoader(
            self.test_dataset, batch_size=1, shuffle=True)

        for data in self.train_data.dataset:
            data = data.to(device)

        for data in self.test_data.dataset:
            data = data.to(device)

    def train_step(self, epoch) -> int:
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        total_loss = 0
        self.model.train()
        for data in self.train_data:
            # print out the filename for the DataBatch
            filename = data.filename[0]
            self.optimizer.zero_grad()
            out = self.model(data)

            # reshape y to be 1x3. -1 infers the size of the dimension from the tensor
            loss = F.l1_loss(out, data.y.view(1, -1))
            total_loss = total_loss + loss
            loss.backward()
            self.optimizer.step()

            if filename not in self.itemized_train_loss:
                self.itemized_train_loss[filename] = []

            y_np = data.y.detach().numpy()
            o_np = out[0].detach().numpy()
            self.itemized_train_loss[filename].append(
                {
                    "epoch": epoch,
                    "agg_loss": loss.detach().numpy(),
                    "loss": [float(x) for x in [y_np[0] - o_np[0], y_np[1] - o_np[1], y_np[2] - o_np[2]]],
                    "output": o_np,
                    "expected": [float(x) for x in y_np],
                }
            )

        return (total_loss / len(self.train_data.dataset))

    def eval_step(self, epoch) -> int:
        self.model.eval()
        total_loss = 0
        for data in self.test_data:
            pred = self.model(data)
            # reshape y to be 1x3. -1 infers the size of the dimension from the tensor
            loss = F.l1_loss(pred, data.y.view(1, -1))
            total_loss = total_loss + loss
            filename = data.filename[0]

            if filename not in self.itemized_test_loss:
                self.itemized_test_loss[filename] = []

            y_np = data.y.detach().numpy()
            o_np = pred[0].detach().numpy()
            self.itemized_test_loss[filename].append(
                {
                    "epoch": epoch,
                    "agg_loss": loss.detach().numpy(),
                    "output": o_np,
                    "loss": [float(x) for x in [y_np[0] - o_np[0], y_np[1] - o_np[1], y_np[2] - o_np[2]]],
                    "expected": [float(x) for x in y_np],
                }
            )

        return (total_loss / len(self.test_data.dataset))

    def train_and_evaluate(self):
        epochs = int(get_config('epochs', 25))
        for epoch in range(epochs):
            train_loss = self.train_step(epoch)
            eval_loss = self.eval_step(epoch)

            self.mean_loss.append(train_loss)
            self.mean_eval_loss.append(eval_loss)

            print("Training and eval for epoch:", epoch, "/ train loss:", train_loss.detach().numpy(), "/ eval loss:", eval_loss.detach().numpy())

        print("Final training loss", self.mean_loss[-1])
        print("Final evaluation loss", self.mean_eval_loss[-1])

        timestr = time.strftime("%Y%m%d_%H%M%S")
        model_type = get_config("model_type").lower()

        with open(os.path.join("outputs", f"{timestr}_{model_type}_{epochs}_test_loss.json"), "w") as f:
            json.dump(self.itemized_test_loss, f, cls=NumpyEncoder)

        with open(os.path.join("outputs", f"{timestr}_{model_type}_{epochs}_train_loss.json"), "w") as f:
            json.dump(self.itemized_train_loss, f, cls=NumpyEncoder)

        # write a csv file with the last value in each object of the loss
        loss_objects = {
            "train": self.itemized_train_loss,
            "test": self.itemized_test_loss
        }
        for loss_name, loss_object in loss_objects.items():
            with open(os.path.join("outputs", f"{timestr}_{model_type}_{epochs}_{loss_name}_loss.csv"), "w") as f:
                f.write("filename,agg_loss,loss,output,expected\n")
                for filename, data in loss_object.items():
                    last = data[-1]
                    agg_loss = last["agg_loss"]
                    loss = "|".join([str(x) for x in last["loss"]])
                    output = "|".join([str(x) for x in last["output"]])
                    expected = "|".join([str(x) for x in last["expected"]])
                    f.write(f"{filename},{agg_loss},{loss},{output},{expected}\n")


if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train_and_evaluate()

    plot_loss(trainer.mean_loss, trainer.mean_eval_loss)
