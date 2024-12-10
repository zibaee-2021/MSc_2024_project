import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_train_val_errors_per_epoch(path_losses_txt: str, include_val_in_plot=True) -> None:
    assert os.path.exists(path_losses_txt), (f"The file `{path_losses_txt}` is missing based on this relative path "
                                             f"from this cwd={os.getcwd()}.")
    losses_per_epoch = np.loadtxt(path_losses_txt, delimiter=",")
    epochs, train_errors, val_errors = losses_per_epoch[:, 0].astype(int), losses_per_epoch[:, 1], losses_per_epoch[:, 2]
    _, ax = plt.subplots()
    padding = 0.1  # to counter the effect of the plot cutting out half of the dots for the last epoch
    # Set extended x-axis limits
    ax.set_xlim(min(epochs), max(epochs) + padding)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.scatter(epochs, train_errors, label='train', color='blue', s=10)
    ax.plot(epochs, train_errors, color='lightblue')
    if include_val_in_plot:
        ax.scatter(epochs, val_errors, label='val', color='red', s=10)
        ax.plot(epochs, val_errors, color='salmon')
    plt.xlabel('epoch')
    plt.ylabel(f'error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    ax.legend(frameon=False, facecolor='none', edgecolor='none')
    plt.show()


if __name__ == '__main__':
    plot_train_val_errors_per_epoch(path_losses_txt='losses_per_epoch_9Dec.txt', include_val_in_plot=False)
