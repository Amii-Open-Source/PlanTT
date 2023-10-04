"""
Authors: Nicolas Raymond

Description: Stores constant indicating the paths leading to important directories within the project
             or pre-trained models on HuggingFace.
"""

from os.path import dirname, join


# Project paths
CURRENT_PROJECT: str = dirname(dirname(__file__))
CHECKPOINTS: str = join(CURRENT_PROJECT, 'checkpoints')
TRAINED_MODELS: str = join(CURRENT_PROJECT, 'trained_models')
SRC: str = join(CURRENT_PROJECT, 'src')
MODELS: str = join(SRC, 'models')
REVERSE_COMPLEMENT_MAP: str = join(SRC, 'data', 'modules', 'reverse_complement_map')

# HuggingFace paths
DDNABERT_HF: str = 'Peltarion/dnabert-minilm-small'
DNABERT_HF: str = 'zhihan1996/DNA_bert'


