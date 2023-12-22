import json
import numpy as np
from utils.config import config as get_config

from matplotlib.pylab import plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_loss(train_loss, test_loss):
    l_train = []
    l_test = []

    for loss in train_loss:
        value = loss.detach().numpy()
        if value > 1:
            value = 2
        l_train.append(value)

    for loss in test_loss:
        value = loss.detach().numpy()
        if value > 5:
            value = 5
        l_test.append(value)

    epochs = range(1, len(l_train) + 1)

    # Plot
    plt.plot(epochs, l_train, label="Training")
    plt.plot(epochs, l_test, label="Testing")

    # Add in a title and axes labels
    model_type = get_config("model_type")
    plt.title(f"Train and Test Loss - {model_type}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Display the plot
    plt.legend(loc="best")
    plt.show()
