import os.path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use("TkAgg")
matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator, MultipleLocator


def plot_train_val_errors_per_epoch(path_losses_txt: str, include_train: bool, include_val: bool) -> \
        None:
    assert os.path.exists(path_losses_txt), (f"The file `{path_losses_txt}` is missing based on this relative path "
                                             f"from this cwd={os.getcwd()}.")
    losses_per_epoch = np.loadtxt(path_losses_txt, delimiter=",")
    epochs, train_errors, val_errors = losses_per_epoch[:, 0].astype(int), losses_per_epoch[:, 1], losses_per_epoch[:, 2]
    fig, ax = plt.subplots(figsize=(8, 10))
    padding = 0.1
    ax.set_xlim(min(epochs), max(epochs) + padding)  # prevent cropping out final epoch dots
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if include_train:
        ax.scatter(epochs, train_errors, label='train', color='blue', s=30)
        ax.plot(epochs, train_errors, color='lightblue')
        # Plot mean of 100 epochs for smoother illustration of trend of loss curves
        bin_size = 100
        bins = len(epochs) // bin_size
        if bins > 0:
            epoch_bins = epochs[:bins * bin_size].reshape(-1, bin_size)
            epoch_means = epoch_bins.mean(axis=1)
            train_error_bins = train_errors[:bins * bin_size].reshape(-1, bin_size)
            train_error_means = train_error_bins.mean(axis=1)
            ax.plot(epoch_means, train_error_means, color='black', linewidth=3)

    if include_val:
        ax.scatter(epochs, val_errors, label='validation', color='red', s=30)
        ax.plot(epochs, val_errors, color='salmon')
        bin_size = 100
        bins = len(epochs) // bin_size
        if bins > 0:
            epoch_bins = epochs[:bins * bin_size].reshape(-1, bin_size)
            epoch_means = epoch_bins.mean(axis=1)
            val_error_bins = val_errors[:bins * bin_size].reshape(-1, bin_size)
            val_error_means = val_error_bins.mean(axis=1)
            ax.plot(epoch_means, val_error_means, color='black', linewidth=3)

    plt.xlabel('epoch', fontsize=14)
    plt.ylabel(f'error', fontsize=14)

    ax.tick_params(axis='both', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.legend()

    if include_train and include_val:
        ax.xaxis.set_major_locator(MultipleLocator(400))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        plt.ylim(top=500, bottom=0)
    elif include_train:
        ax.xaxis.set_major_locator(MultipleLocator(400))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        plt.ylim(top=35, bottom=20)
    elif include_val:
        ax.xaxis.set_major_locator(MultipleLocator(400))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        plt.ylim(top=500, bottom=200)

    fig.subplots_adjust(left=0.2, right=0.9)

    if include_train and include_val:
        plt.legend()
        ax.legend(
            loc='center left',  # Align legend to left of anchor point
            bbox_to_anchor=(1.05, 0.5),  # Move legend right (1.1 means 10% outside plot)
            handletextpad=0,  # space between marker dot and word
            borderpad=0,  # padding around legend box
            fontsize=14,
            facecolor='none',
            edgecolor='none',
            frameon=False
        )
    x_ticks = list(range(0, 2000, 400)) + [1999]  # Adding 1999 explicitly
    ax.set_xticks(x_ticks)
    ax.set_xlim(min(epochs), max(epochs))
    plt.tight_layout()

    if include_train and include_val:
        plt.savefig('loss_plot_18Dec.png', bbox_inches='tight')
    elif include_train:
        plt.savefig('train_loss_plot_18Dec.png', bbox_inches='tight')
    elif include_val:
        plt.savefig('val_loss_plot_18Dec.png', bbox_inches='tight')
    else:
        pass
    plt.show()


if __name__ == '__main__':
    # For plots of both the training and validation losses overlapping on same plot, select True for both.
    # Otherwise select True only for the plot you want a close up of, False for the other:
    plot_train_val_errors_per_epoch(path_losses_txt='losses_per_epoch_18Dec.txt',
                                    include_train=True,
                                    include_val=True)
