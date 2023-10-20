"""
Author: Nicolas Raymond

Description: Stores the procedure to create one-hot encodings and 
             6-mer tokens from raw data samples.
"""
from os.path import join
from pandas import read_csv
from pickle import dump
from torch import cat, stack, LongTensor, ones, Tensor
from transformers import BertTokenizer

# Project imports
from settings.paths import DATA, DNABERT_HF
from src.data.modules import preprocessing as p

# Script execution
if __name__ == '__main__':

    for dataset_type in ['training', 'validation']:

        # Load the raw samples
        df = read_csv(join(DATA, f'{dataset_type}_samples.csv'))

        # Extract the targets
        targets = Tensor(df['targets'].to_list())

        # Create dictionary of tensors to store the processed data
        encoded_seq: dict[str, Tensor] = {}
        tokenized_seq: dict[str, Tensor] = {}

        for column in ['seq_a', 'seq_b']:

            # Extract sequences
            sequences = df[column].to_list()

            # Generate the one-hot encodings
            one_hot = stack([p.nucleotide_to_onehot(seq) for seq in sequences])
            encoded_seq[column] = one_hot.permute(0, 2, 1)

            # Initialize a tokenizer for to generate 6-mer tokens
            tokenizer = BertTokenizer.from_pretrained(f'{DNABERT_HF}_6')

            for i, seq in enumerate(sequences):

                # Convert the sequence to 6-mers and tokenize it
                # str -> list[int]
                seq = tokenizer.convert_tokens_to_ids(p.seq_to_kmer_tokens(seq, 6))

                # Split tokenized sequence into 6 blocks of 510 tokens
                # list[int] -> list[list[int]]
                seq = p.split_tokens(seq, max_length=510, overlap=0)

                # Pad last block of length shorter than 510
                # list[list[int]] -> LongTensor
                seq = LongTensor(p.add_padding(tokens_subseqs=seq,
                                               padding_token_id=tokenizer.vocab['[PAD]'],
                                               max_length=510,
                                               padding_option='right'))

                # Add [CLS] and [SEP] tokens at the beginning and the end of all blocks
                seq = cat(((ones(len(seq), 1)*tokenizer.vocab['[CLS]']).long(),
                           seq, (ones(len(seq), 1)*tokenizer.vocab['[SEP]']).long()), dim=1)

                # Update the sequence
                sequences[i] = seq

            # Save the tokenized sequences
            tokenized_seq[column] = stack(sequences)

        # Save the processed data
        with open(join(DATA, f'encoded_{dataset_type}_samples.pkl'), 'wb') as file:
            dump([encoded_seq['seq_a'], encoded_seq['seq_b'], targets], file)

        with open(join(DATA, f'tokenized_{dataset_type}_samples.pkl'), 'wb') as file:
            dump([tokenized_seq['seq_a'], tokenized_seq['seq_b'], targets], file)
