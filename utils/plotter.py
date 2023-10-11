import json
import numpy as np

from matplotlib.pylab import plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_loss(data, label=""):
    losses = [l.item() for l in data]
    epochs = range(1, len(losses) + 1)

    # Plot
    plt.plot(epochs, losses, label=label)

    # Add in a title and axes labels
    plt.title(label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Display the plot
    plt.legend(loc="best")
    plt.show()
