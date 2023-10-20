"""
Author: Nicolas Raymond

Description: This file stores a program returning a list of singe-base edits
             that increase the rank expression of a gene based on its sequence.
"""
from os.path import join
from torch import load, device
from torch.cuda import is_available
from warnings import warn

# Project imports
from settings.paths import TRAINED_MODELS
from src.data.modules.preprocessing import nucleotide_to_onehot
from src.models.cnn import CNN1D
from src.models.head import SumHead
from src.models.plantt import PlanTT
from src.utils.gene_editing import get_editing_strategy

# Batch size used to generate the embeddings of the single-base variations
BATCH_SIZE: int = 1000


def validate_sequence_format(seq: str) -> None:
    """
    Validates if a DNA sequence passed a as string meets specific criteria.
 
    The sequence is considered valid if it has exactly 1500 characters
    and each character is either 'A', 'C', 'G', 'T', 'N', or 'X'.

    Args:
        seq (str): DNA sequence
    """
    # Check if the string has exactly 1500 characters
    if len(seq) != 1500:
        raise ValueError('The sequence must contain 1500 characters')

    # Check if each character in the string is one of the valid characters
    valid_characters = {'A', 'C', 'G', 'T', 'N', 'X'}
    for char in seq:
        if char not in valid_characters:
            raise ValueError(f'Characters provided must be among {valid_characters}')

# Execution of the program
if __name__ == '__main__':

    # Set the device
    if is_available():
        dev = device('cuda')
    else:
        warn('No GPU was found, expect the process to be slower.')
        dev = device('cpu')

    # Ask for the promoter sequence
    promoter = input('\nEnter the 1500bp sequence representing the promoter region: ').upper()
    validate_sequence_format(promoter)

    # Ask for the terminator sequence
    terminator = input('\nEnter the 1500bp sequence representing the terminator region: ').upper()
    validate_sequence_format(terminator)

    # Ask for the budget of single-base edits
    budget = int(input('\nEnter the maximal number of single-base edits that can be applied: '))

    # Concatenate both sequence and apply one-hot encoding
    encoded_seq = nucleotide_to_onehot(seq=promoter+terminator).permute(1, 0)

    # Load PlanTT-CNN model
    plantt = PlanTT(tower=CNN1D(seq_length=3000, feature_dropout=0, dropout=0),
                    head=SumHead(regression=True))

    plantt.load_state_dict(load(join(TRAINED_MODELS, 'planttcnn.pt')))

    # Find the editing strategy
    edits = get_editing_strategy(plantt=plantt,
                                 seq=encoded_seq,
                                 dev=dev,
                                 max_edits=budget,
                                 batch_size=BATCH_SIZE)

    # Print the results
    print('\nList of edits found: \n')
    total = 0
    for i, e in enumerate(edits):
        print(f'{i}. Modification: {e[0]}, Position: {e[1]}, Rank expression improvement: {e[2]}')
        total += e[2]

    print(f'\nExpected improvement in rank expression: {total}\n')

    print('\nSee the following files to visualize the results:\n')
    for i in range(budget):
        print(f'edit_{i}.pdf')
    print('\n')
