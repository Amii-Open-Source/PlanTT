"""
Authors: Nicolas Raymond

Description: Create a dictionary mapping each token id of a k-mer vocabulary to the id of the token
             associated to its reverse complement.
"""
import sys

from argparse import ArgumentParser
from json import dump
from os.path import abspath, join, pardir
from transformers import BertTokenizer


sys.path.append(abspath(join(__file__, *[pardir]*4)))
from settings.paths import DNABERT_HF, REVERSE_COMPLEMENT_MAP
from src.data.modules.preprocessing import get_reverse_complement

# Definition of the arguments parsing function
def get_args():
    """
    Retrieves the arguments given in the terminal to run the script.
    """
    parser = ArgumentParser(usage='python create_reverse_complement_map.py -k [...]',
                            description='Create a mapping between each token in the vocabulary and its reverse complement')
    
    # Folder path
    parser.add_argument('-k', '--k', type=int, default=6, choices=[3, 4, 5, 6],
                        help='K value associated to the k-mer vocabulary.')
    
    return parser.parse_args()


# Execution of the script
if __name__ == '__main__':
    
    # Retrieve the arguments
    args = get_args()
    
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'{DNABERT_HF}_{args.k}')
    
    # Initialize a ID to reverse complement ID map
    id_to_reverse_comp_id = {}
    
    for token, id in tokenizer.vocab.items():
        
        # If it is not a special token
        if token not in ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']:
            
            # Get the id of its reverse complement
            id_to_reverse_comp_id[int(id)] = int(tokenizer.vocab[get_reverse_complement(token)])
            
        else:
            
            # Map the id to itself
            id_to_reverse_comp_id[int(id)] = int(id)
    
    with open(join(REVERSE_COMPLEMENT_MAP, 'map_6'), 'w') as file:
        dump(id_to_reverse_comp_id, file, indent=True)
            
            
            
            
            





