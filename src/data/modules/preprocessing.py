"""
Authors: Nicolas Raymond
         Fatima Davelouis

Description: Stores all the functions related to DNA data preprocessing.
"""
from torch import Tensor

NUC2ONEHOT_MAPPING = {"A": [1, 0, 0, 0, 0],
                      "C": [0, 1, 0, 0, 0],
                      "G": [0, 0, 1, 0, 0],
                      "T": [0, 0, 0, 1, 0],
                      "N": [0, 0, 0, 0, 1],
                      "X": [0, 0, 0, 0, 1]
                      }

ONEHOT2NUC_MAPPING = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'NX'}


def get_reverse_complement(seq: str) -> str:
    """
    Reverses the order of the characters and switches A <-> T and C <-> G.
    Other characters are left untouched.
    
    Args:
        seq (str): sequence of nucleotides

    Returns:
        str: reverse complement of the sequence
    """
    return seq[::-1].translate(str.maketrans('ACGT', 'TGCA'))

def nucleotide_to_onehot(seq: str) -> Tensor:
    """
    Convert each nucleotide of a sequence to its one-hot representation.
    
    Args:
        seq (str): sequence of nucleotides

    Returns:
        Tensor: sequence represented using one-hot encodings (SEQ LENGTH, 5)
    """
    return Tensor([NUC2ONEHOT_MAPPING.get(nucleotide, [0, 0, 0, 0, 1]) for nucleotide in seq])


def filter_seq(seq: str) -> str:
    """
    Removes the 'N' and the 'X' from a sequence of nucleotides.
    
    Args:
        seq (str): sequence of nucleotides.

    Returns:
        str: same sequence without any 'N' and 'X'.
    """
    return seq.translate(str.maketrans('', '', 'NX'))


def seq_to_kmer_tokens(seq: str,
                       k: int = 6,
                       stride: int = 1) -> list[str]:
    """
    Takes a sequence of nucleotides and generate a list of k-mers.
    To avoid the inclusion of an incomplete k-mer, the iterative creation process stops
    before stride*i + k > len(seq).
    
    Args:
        seq (str): sequence of nucleotides.
        k (int, optional): length of k-mers.
                           Default to 6.
        stride (int, optional): step size in between each k-mers. 
                                If stride < k, than two contiguous k-mers will share
                                (k - stride) nucleotides. 
                                Default to 1.
    Returns:
        list[str]: list of k-mers of length floor((len(seq) - k)/stride + 1)
    """
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]


def split_tokens(tokens: list[str],
                 max_length: int = 512,
                 overlap: int = 256) -> list[list[int]]:
    """
    Extract subsequences of tokens from a list of tokens.
    IMPORTANT DETAIL: tokens[i:i+max_length] == tokens[i:min(len(tokens), i+max_length)]
    
    Args:
        tokens (list[str]): list of tokens.
        max_length (int, optional): maximum number of tokens included within each subsequence.
                                    Defaults to 512.
        overlap (int, optional): number of tokens shared by two contiguous subsequence. 
                                 Defaults to 256.

    Returns:
        list[list[int]]: list of list of tokens
    """
    return [tokens[i:i+max_length] for i in range(0, len(tokens), max_length - overlap)]


def add_padding(tokens_subseqs: list[list[int]],
                padding_token_id: int,
                max_length: int = 512,
                padding_option: str = 'right') -> list[list[int]]:
    """
    Adds padding to subsequences of tokens of length < max_length.
    Padding is added to the left and the right of any subsequence of tokens.
    
    Args:
        tokens_subseqs (list[list[int]]): list containing list of tokens.
        padding_token_id (int): id associated to the padding token
        max_length (int, optional): maximum number of tokens included within each subsequence.
                                    Default to 512.
        padding_option (str, optional): 'right', 'left' or 'both'.
                                        Default to 'right'.

    Returns:
        list[list[int]]: list containing list of tokens.
    """
    for i, subseq in enumerate(tokens_subseqs):

        # If padding needs to be added
        subseq_length = len(subseq)

        if subseq_length < max_length:

            # Check validity of padding option
            if padding_option not in ['right', 'left', 'both']:
                return ValueError("padding_option must be either 'right', 'left' of 'both'")

            elif padding_option == 'both':

                # Generate the padding
                padding = [padding_token_id]*((max_length - subseq_length)//2)

                # If the current the length of the current subseq is even
                if subseq_length % 2 == 0:

                    # Add equal amount of padding on left and right
                    tokens_subseqs[i] = padding + subseq + padding

                else:

                    # Add more padding on the right than the left
                    tokens_subseqs[i] = padding + subseq + padding + [padding_token_id]

            else:

                # Generate the padding
                padding = [padding_token_id]*(max_length - subseq_length)

                if padding_option == 'left':

                    # Add padding on left
                    tokens_subseqs[i] = padding + subseq

                else:

                    # Add padding on right
                    tokens_subseqs[i] = subseq + padding

    return tokens_subseqs
