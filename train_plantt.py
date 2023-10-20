"""
Author: Nicolas Raymond

Description: This file stores the procedure to train plantt.
"""

from argparse import ArgumentParser
from datetime import datetime
from json import dump, load
from os import makedirs
from os.path import join
from pickle import load as pkload
from time import time
from torch import device
from torch.backends import cudnn
from torch.cuda import is_available, set_per_process_memory_fraction, device_count

# Project imports
from settings.paths import RECORDS
from src.data.modules.dataset import OCMDataset
from src.models.cnn import CNN1D
from src.models.head import SumHead
from src.models.mlm import DDNABERT, DNABERT
from src.models.plantt import PlanTT
from src.optimization.training import OCMTrainer
from src.utils import metrics as m
from src.utils.analysis import save_progress_figure
from src.utils.loss import MSE
from src.utils.reproducibility import SEED, set_seed

# Definition of the argument parsing function
def get_settings():
    """
    Retrieves the settings given in the terminal to run the script.
    """
    parser = ArgumentParser(usage='python train_plantt.py',
                            description='Retrieves the settings for the training.')

    # Model selection
    parser.add_argument('-tower', '--tower', type=str,
                        choices=['cnn', 'ddnabert', 'dnabert'], default='cnn',
                        help='Choice of the tower architecture used for the training.')

    # Data paths
    parser.add_argument('-t_data', '--training_data', type=str,
                        help='Path leading to the pickle file with the training data.')

    # Data paths
    parser.add_argument('-v_data', '--valid_data', type=str,
                        help='Path leading to the pickle file with the validation data.')

    # Tokenize data
    parser.add_argument('-tokens', '--tokens', default=False, action='store_true',
                        help='If provided, the data is expected to contain tokenized 6-mers.')

    # Training and validation batch sizes
    parser.add_argument('-tbs', '--train_batch_size', type=int, default=32,
                        help='Training batch size. Default to 32')

    parser.add_argument('-vbs', '--valid_batch_size', type=int, default=32,
                        help='Validation batch size. Default to 32')

    # Initial learning rate
    parser.add_argument('-lr', '--lr', type=float, default=5e-5,
                        help='Initial learning rate. Default to 5e-5.')

    # Max epochs
    parser.add_argument('-max_e', '--max_epochs', type=int, default=200,
                        help='Maximum number of epochs. Default to 200.')

    # Patience
    parser.add_argument('-patience', '--patience', type=int, default=20,
                        help='Number of epochs without improvement allowed \
                              before stopping the training. Only the weights associated to \
                              the best validation score are kept at the end of the training. \
                              Default to 20.')

    # Milestones
    parser.add_argument('-ms', '--milestones', nargs='*', type=int, default=None,
                        help='Epochs after which the learning rate is multiplied by \
                              a factor of gamma (ex. 50 60 70). When set to None, \
                              the learning rate is multiplied by a factor of gamma \
                              every 15 epochs, starting from epoch 75th. \
                              Default to None.')

    # Gamma
    parser.add_argument('-gamma', '--gamma', type=float, default=0.75,
                        help='Constant multiplying the learning rate at each milestone. \
                              Default to 0.75')
    # Weight decay
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2,
                        help='Weight decay (L2 penalty coefficient). Default to 1e-2.')
    # Dropout
    parser.add_argument('-p', '--dropout', type=float, default=0.0,
                        help='Probability of the elements to be zeroed following \
                              convolution layers (Applies to plantt-cnn only). \
                              Default to 0.0.')

    # Freeze method for pre-trained transformer model
    parser.add_argument('-fm', '--freeze_method', type=str, default=None,
                        choices=['all', 'keep_last', None],
                        help='Freeze method used if a pre-trained language model is selected. \
                              If "all", all layers are frozen. \
                              If "keep_last", all layers except the last one are frozen.\
                              If None, all layers remain unfrozen. \
                              Default to None')

    # Device ID
    parser.add_argument('-dev', '--device_id', type=int, default=None,
                        help='Cuda device ID. Default to None.')

    # Memory fraction
    parser.add_argument('-memory', '--memory_frac', type=float, default=1,
                        help='Percentage of device allocated to the training. \
                              Default to 1.')
    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=SEED,
                        help=f'Seed value used for training reproducibility. \
                             Default to {SEED}.')

    return parser.parse_args()


# Execution of the script
if __name__ == '__main__':

    # Retrieve environment settings
    SETTINGS: dict = vars(get_settings())

    # Check for GPU availability
    if is_available():
        if SETTINGS['device_id'] is not None:
            DEVICE = device(f"cuda:{SETTINGS['device_id']}")
        else:
            DEVICE = device('cuda:0')
    else:
        raise ValueError('Since models are trained with autocast and float16, \
                          a GPU is required for the experiment.')

    # Set GPU memory allocation
    if 0 < SETTINGS['memory_frac'] < 1:
        set_per_process_memory_fraction(fraction=SETTINGS['memory_frac'], device=DEVICE)

    # Create PlanTT tower
    if SETTINGS['tokens']:
        if SETTINGS['tower'] == 'ddnabert':
            tower = DDNABERT(freeze_method=SETTINGS['freeze_method'])
        elif SETTINGS['tower'] == 'dnabert':
            tower = DNABERT(freeze_method=SETTINGS['freeze_method'])
        else:
            raise ValueError('PlanTT-CNN is not compatible with tokenized data.')
    else:
        if SETTINGS['tower'] == 'cnn':
            tower = CNN1D(seq_length=3000, dropout=SETTINGS['dropout'])
        else:
            raise ValueError('The PlanTT tower selected is only compatible with tokenized data.')

    # Create PlanTT model
    plantt = PlanTT(tower=tower, head=SumHead(regression=True))

    # Load the data
    with open(SETTINGS['training_data'], 'rb') as file:
        x_train_a, x_train_b, y_train = pkload(file)
    with open(SETTINGS['valid_data'], 'rb') as file:
        x_valid_a, x_valid_b, y_valid = pkload(file)

    # Create the datasets
    train_set = OCMDataset(seq_a=x_train_a,
                           seq_b=x_train_b,
                           labels=y_train,
                           species=None,
                           tokens=SETTINGS['tokens'])

    valid_set = OCMDataset(seq_a=x_valid_a,
                           seq_b=x_valid_b,
                           labels=y_valid,
                           species=None,
                           tokens=SETTINGS['tokens'])

    # Create folders to store the results
    EXPERIMENT_FOLDER: str = join(RECORDS, datetime.now().strftime('%d_%m_%Y_%H:%M:%S'))
    FIGURES_FOLDER: str = join(EXPERIMENT_FOLDER, "figures")
    makedirs(FIGURES_FOLDER)

    # Save the settings of the experiment
    with open(join(EXPERIMENT_FOLDER, 'settings.json'), 'w', encoding="utf-8") as file:
        dump(SETTINGS, file, indent=True)

    # Set the seed for reproducibility
    cudnn.deterministic = True
    set_seed(seed_value=SETTINGS['seed'], n_gpu=1)

    # Initialize the metrics to track during training and validation
    metrics = [m.Accuracy(), m.BalancedAccuracy(), m.F1Score(),
               m.Precision(), m.Recall(), m.MeanAbsoluteError(),
               m.SpearmanRankCorrelation(), m.RSquared(),
               m.MeanAbsolutePercentageError(), m.RootMeanSquaredError()]

    # Initialize the trainer
    trainer = OCMTrainer(loss_function=MSE())

    # Train the model
    start = time()
    model, last, best = trainer.train(model=plantt,
                                      dev=DEVICE,
                                      datasets=(train_set, valid_set),
                                      batch_sizes=(SETTINGS['train_batch_size'],
                                                   SETTINGS['valid_batch_size']),
                                      metrics=metrics,
                                      lr=SETTINGS['lr'],
                                      max_epochs=SETTINGS['max_epochs'],
                                      patience=SETTINGS['patience'],
                                      weight_decay=SETTINGS['weight_decay'],
                                      record_path=join(EXPERIMENT_FOLDER,
                                                       f"plantt_{SETTINGS['tower']}"),
                                      scheduler_milestones=SETTINGS['milestones'],
                                      scheduler_gamma=SETTINGS['gamma'],
                                      return_epochs=True)

    # Save the training time and the number of epochs
    results = {'training_time': round((time() - start)/60, 2),
               'total_epochs': last,
               'best_epoch': best}

    # Recover the final training and validation metrics obtained
    JSON_PATH = join(EXPERIMENT_FOLDER, f"plantt_{SETTINGS['tower']}.json")
    with open(JSON_PATH, 'rb') as file:
        data = load(file)
        train_records, valid_records = data['train'], data['valid']

    # Save the scores
    results['train_metrics'] = {met.name: round(train_records[met.name][best], 4)
                                for met in metrics}

    results['valid_metrics'] = {met.name: round(valid_records[met.name][best], 4)
                                for met in metrics}

    # Save a summary of the training
    with open(join(EXPERIMENT_FOLDER, 'summary.json'), 'w', encoding='utf-8') as file:
        dump(results, file, indent=True)

    # Create figures for the progression of each metric during the training
    for met in train_records.keys():
        save_progress_figure(train_scores=train_records[met],
                             valid_scores=valid_records[met],
                             metric_name=met,
                             figure_path=join(FIGURES_FOLDER, f'{met}.pdf'))
