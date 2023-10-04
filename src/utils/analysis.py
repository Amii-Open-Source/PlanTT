"""
Authors: Nicolas Raymond

Description: Stores function dedicated to the analysis of experiment results.
"""

from matplotlib import pyplot as plt
from numpy import array, arange
from pandas import read_csv
from seaborn import heatmap
from scipy.stats import ttest_rel


def save_progress_figure(train_scores: array,
                         valid_scores: array,
                         metric_name: str,
                         figure_path: str) -> None:
    """
    Saves a figure illustrating the progress of a metric over the training epochs.
    
    Args:
        train_scores (list[float]): list or array with scores obtained on the training set during each epoch.
        valid_scores (list[float]): list or array with scores obtained on the validation set during each epoch.
        metric_name (str): name of the metric.
        figure_path (str): path used to save the figure created.
    """
    # Save the number of epochs
    nb_epochs = len(train_scores)
    
    # Check if the given scores list are of the same length
    if len(valid_scores) != nb_epochs:
        raise Exception('train_scores and valid_scores are not of the same length')
    
    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax1.plot(range(nb_epochs), train_scores)
    ax2.plot(range(nb_epochs), valid_scores, 'tab:orange')
    
    # Set the labels
    for ax in (ax1, ax2):
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        
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
    Generates a heatmap showing the predicted impact of single-base edits within a window of 51bp around the argmax.

    Args:
        predictions (array): mRNA abundance difference predictions (4, 3000)
        arg_max (int): index that will be at the center of the window.
        iteration (int): id to help identify the figure.
    """
    plt.rcParams.update({'font.size': 5, 'font.family': 'serif'})
    _, ax = plt.subplots(figsize=(5,1))
    start, stop = arg_max - 25, arg_max + 26
    heatmap(predictions[:, start:stop], ax=ax, linewidths=0.2, square=True, cmap='viridis', mask=(predictions[:, start:stop] == 0),
            cbar_kws = dict(use_gridspec=False, location="top", fraction=0.22, aspect=120, shrink=0.45, label='mRNA abundance difference'))
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
    
    
def execute_paired_ttest(score_csv_a: str,
                         score_csv_b: str) -> dict[str, float]:
    """
    Compares the averages of cross-validation metrics obtained by two models.
    
    For each test, we have:
    Null hypothesis (H0): The two repeated samples have identical average 
    Alternate hypothesis (H1): The means of the distributions underlying the samples are unequal.
    
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html.
    
    Args:
        score_csv_a (str): path of the file containing test scores of model a.
        score_csv_b (str): path of the file containing test scores of model a.

    Returns:
        dict[str, float]: metric names and associated p-values
    """

    # Load the content of the csv files into pandas data frame
    df_a = read_csv(score_csv_a, index_col='Fold').drop('mean')
    df_b = read_csv(score_csv_b, index_col='Fold').drop('mean')
    
    # Calculate the p-values using
    return {metric: f'{ttest_rel(df_a[metric].to_numpy(), df_b[metric].to_numpy()).pvalue:.4f}' for metric in df_a.columns}
    
    
