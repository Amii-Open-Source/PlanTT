"""
Authors: Nicolas Raymond

Description: Stores functions dedicated to the analysis of experiment results.
"""

from matplotlib import pyplot as plt
from numpy import array, arange
from seaborn import heatmap


def save_progress_figure(train_scores: array,
                         valid_scores: array,
                         metric_name: str,
                         figure_path: str) -> None:
    """
    Saves a figure illustrating the progress of a metric over the training epochs.
    
    Args:
        train_scores (array): scores obtained on the training set during each epoch.
        valid_scores (array): scores obtained on the validation set during each epoch.
        metric_name (str): name of the metric.
        figure_path (str): path used to save the figure created.
    """
    # Save the number of epochs
    nb_epochs = len(train_scores)

    # Check if the given scores list are of the same length
    if len(valid_scores) != nb_epochs:
        raise ValueError('train_scores and valid_scores are not of the same length')

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax1.plot(range(nb_epochs), train_scores)
    ax2.plot(range(nb_epochs), valid_scores, 'tab:orange')

    # Set the labels
    for ax in (ax1, ax2):
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)

    ax1.set_title('Training')
    ax2.set_title('Validation')

    # Hide the labels for the right panel
    for ax in (ax1, ax2):
        ax.label_outer()

    # Adjust the figure and save it
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def save_editing_heatmap(predictions: array,
                         arg_max: int,
                         iteration: int) -> None:
    """
    Generates a heatmap showing the predicted impact of single-base
    edits within a window of 51bp around the argmax.

    Args:
        predictions (array): mRNA abundance difference predictions (4, 3000)
        arg_max (int): index that will be at the center of the window.
        iteration (int): id to help identify the figure.
    """
    plt.rcParams.update({'font.size': 5, 'font.family': 'serif'})
    _, ax = plt.subplots(figsize=(5,1))
    start, stop = arg_max - 25, arg_max + 26

    # Heatmap creation
    heatmap(predictions[:, start:stop],
            ax=ax, linewidths=0.2,
            square=True,
            cmap='viridis',
            mask=(predictions[:, start:stop] == 0),
            cbar_kws = dict(use_gridspec=False, location="top",
                            fraction=0.22, aspect=120, shrink=0.45,
                            label='mRNA abundance rank difference'))

    # Axes settings
    ax.set_xticks(arange(0.5, (stop-start)+0.5, 5))
    ax.set_xticklabels(list(range(start, stop, 5)))
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['A', 'C', 'G', 'T'])
    ax.tick_params(axis='y', width=0.2, labelrotation=45, length=1)
    ax.tick_params(axis='x', width=0.2, labelrotation=0, pad=1.5, length=1)
    ax.set_xlabel('Nucleotide position')
    ax.set_ylabel('Nucleotide')

    plt.savefig(f'edit_{iteration}.pdf')
    plt.close()
