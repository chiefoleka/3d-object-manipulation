from matplotlib.pylab import plt

def plot_loss(data, label=''):
    losses = [l.item() for l in data]
    epochs = range(1, len(losses) + 1)

    # Plot
    plt.plot(epochs, losses, label=label)

    # Add in a title and axes labels
    plt.title(label)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display the plot
    plt.legend(loc='best')
    plt.show()

def _plot_eval(expected, output):
    # Calculate the mean absolute value
    pass
