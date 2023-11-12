import json
import numpy as np

from matplotlib.pylab import plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_loss(train_loss, test_loss):
    l_train = [l.item() for l in train_loss]
    l_test = [l.item() for l in test_loss]
    epochs = range(1, len(l_train) + 1)

    # Plot
    plt.plot(epochs, l_train, label="Training")
    plt.plot(epochs, l_test, label="Testing")

    # Add in a title and axes labels
    plt.title("Train and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Display the plot
    plt.legend(loc="best")
    plt.show()
