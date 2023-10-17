"""
Authors: Nicolas Raymond
         Fatima Davelouis
         
Description: Stores functions that enable the reproducibility of the experiments.
"""
from random import seed
from numpy.random import seed as np_seed
from torch import manual_seed
from torch.cuda import manual_seed_all


# Experiment seed value
SEED: int = 1


def set_seed(seed_value: int,
             n_gpu: int) -> None:
    """
    Sets the seed value associated to the main libraries.
    
    Args:
        seed_value (int): seed value.
        n_gpu (int): number of gpu available.
    """
    # Set the seed value using random, numpy and torch libraries
    seed(seed_value)
    np_seed(seed_value)
    manual_seed(seed_value)
    if n_gpu > 0:
        manual_seed_all(seed_value)
