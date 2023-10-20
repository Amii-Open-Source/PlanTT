"""
Authors: Nicolas Raymond

Description: Stores the classes associated to tower architectures
             based on Masked Language Model (MLM).
"""

from torch import load, stack, Tensor
from torch.nn import Module
from transformers import BertForMaskedLM, BertForSequenceClassification, logging

# Project imports
from settings.paths import DDNABERT_HF, DNABERT_HF
from src.models.blocks import CNNBlock1D
from src.models.plantt import Tower


class DDNABERT(Tower):
    """
    Distilled DNABERT6 model with an additional adapter layer to connect to PlanTT.
    """
    def __init__(self, freeze_method: str = None) -> None:
        """
        Loads the distilled version of DNABERT6 and initializes the adapter layer.
        
        freeze_method (str, optional): If 'all', all the layers of the DNABERT6 model are frozen.
                                       If 'keep_last', all layers are frozen except the last.
                                       Otherwise, all layers remain unfrozen.
                                       Default to None.
        """
        # Initialize the adapter layer
        # Note: we cannot assign a module to a private attribute yet
        #       because tower init need to be called first
        adapter_layer = MLMAdapter(chunk_length=512, chunk_depth=384)

        # Call parent constructor
        super().__init__(adapter_layer.out_size)
        self.__adapter_layer = adapter_layer

        # Load distilled DNABERT6 and turn off warning during loading
        logging.set_verbosity_error()
        self.__dnabert = BertForSequenceClassification.from_pretrained(DDNABERT_HF)
        logging.set_verbosity_warning()

        # Freeze the weights
        if freeze_method == 'all':
            for param in self.__dnabert.parameters():
                param.requires_grad = False

        elif freeze_method == 'keep_last':
            for name, param in self.__dnabert.named_parameters():
                if 'layer.5' not in name:
                    param.requires_grad = False
        else:
            pass

    def forward(self, seq: Tensor) -> Tensor:
        """
        Executes a forward pass.
        Since sequences are longer than 512 tokens, each of them is separated
        into multiple chunks of 512 tokens.

        Args:
            seq (Tensor): batch of tokenized nucleotide sequences (BATCH SIZE, NB CHUNK, 512)
            
        Returns:
            Tensor: embeddings of each sequence in the batch (BATCH SIZE, 512)
        """
        # Reshape tensor
        # (BATCH SIZE, NB CHUNK, 512) -> (BATCH SIZE*NB CHUNK, 512)
        seq = seq.reshape(-1, 512)

        # Average all the hidden states of mini dna
        # (BATCH SIZE*NB CHUNK, 512) -> (BATCH SIZE*NB CHUNK, 512, 384)
        seq = stack(self.__dnabert(seq).hidden_states, dim=2).mean(dim=2)

        # Pass embeddings through the adapter layer
        # (BATCH SIZE*NB CHUNK, 512, 384) -> (BATCH SIZE, 512)
        return self.__adapter_layer(seq.permute(0, 2, 1))


class DNABERT(Tower):
    """
    Plain DNABERT-k model with an additional adapter layer to connect to PlanTT.
    """
    def __init__(self,
                 k: int = 6,
                 freeze_method: str = None,
                 pre_trained_weights: str = None) -> None:
        """
        Loads DNABERT-k and initializes the adapter layer.
        
        k (int): Value specifying that the model was trained with k-mer tokens.
        freeze_method (str, optional): If 'all', all the layers of the DNABERT model are frozen.
                                       If 'keep_last', all layers are frozen except the last.
                                       Otherwise, all layers remain unfrozen.
                                       Default to None.

        pre_trained_weights (str, optional): Path from which pre-trained weights
                                             of DNABERT-k must be loaded.
                                             Default to None.
        """
        # Initialize the adapter layer
        # Note: we cannot assign a module to a private attribute yet
        #       because tower init need to be called first)
        adapter_layer = MLMAdapter(chunk_length=512, chunk_depth=768)

        # Call parent constructor
        super().__init__(adapter_layer.out_size)
        self.__adapter_layer: MLMAdapter = adapter_layer

        # Load DNABERT6 architecture (keep model public to allow extraction for pre-training)
        self.dna_bert = BertForMaskedLM.from_pretrained(f'{DNABERT_HF}_{k}')

        # Load weights obtained from pre-training on plant genome
        if pre_trained_weights is not None:
            self.dna_bert.load_state_dict(load(pre_trained_weights))

        # Freeze the weights
        if freeze_method == 'all':
            for param in self.dna_bert.parameters():
                param.requires_grad = False

        elif freeze_method == 'keep_last':
            for name, param in self.dna_bert.named_parameters():
                if 'layer.11' not in name:
                    param.requires_grad = False
        else:
            pass

    def forward(self, seq: Tensor) -> Tensor:
        """
        Executes a forward pass.
        
        Since sequences are longer than 512 tokens, each of them is separated
        into multiple chunks of 512 tokens.

        Args:
            seq (Tensor): batch of tokenized nucleotide sequences (BATCH SIZE, NB CHUNK, 512)
            
        Returns:
            Tensor: embeddings of each sequence in the batch (BATCH SIZE, 512)
        """
        # Reshape tensor
        seq = seq.reshape(-1, 512)  # (BATCH SIZE, NB CHUNK, 512) -> (BATCH SIZE*NB CHUNK, 512)

        # Average all the hidden states
        # (BATCH SIZE*NB CHUNK, 512) -> (BATCH SIZE*NB CHUNK, 512, 768)
        hidden_states = self.dna_bert(input_ids=seq, output_hidden_states=True).hidden_states
        seq = stack(hidden_states, dim=2).mean(dim=2)

        # Pass embeddings through the adapter layer
        # (BATCH SIZE*NB CHUNK, 512, 768) -> (BATCH SIZE, 768)
        return self.__adapter_layer(seq.permute(0, 2, 1))


class MLMAdapter(Module):
    """
    Module in charge of encoding the embeddings generated
    by Masked Language Model (MLM) in a proper format for PlanTT.
    """
    def __init__(self,
                 nb_chunks: int = 6,
                 chunk_length: int = 512,
                 chunk_depth: int = 768,
                 kernel_size: int = 6,
                 stride: int = 6) -> None:
        """
        Sets the protected attributes and the 1D convolution layers.

        Args:
            nb_chunks (int, optional): Number of chunks associated to each complete
                                       nucleotides sequences. 
                                       Default to 6.
            chunk_length (int, optional): Length of each chunk. 
                                          Default to 512.
            chunk_depth (int, optional): Number of channels into each chunk. 
                                         Default to 768.
            kernel_size (int, optional): Kernel size used in the last convolution layer.
                                         Default to 6.
            stride (int, optional): Stride used in the last convolution layer. 
                                    Default to 6
                                    
        Raises:
            ValueError: If the output size expected is not an integer, an error will be raised.
        """
        super().__init__()
        self.__nb_chunks: int = nb_chunks
        self.__chunk_length: int = chunk_length
        self.__bottleneck = CNNBlock1D(in_channels=chunk_depth,
                                       out_channels=1,
                                       kernel_size=1,
                                       stride=1)
        self.__out_size: float = ((nb_chunks*chunk_length) - kernel_size)/stride + 1

        if not self.__out_size.is_integer():
            raise ValueError('Input values must be changed to ensure that \
                             ((NB CHUNKS*CHUNK LENGTH) - KERNEL SIZE)/STRIDE + 1 is an integer')

        self.__out_size: int = int(self.__out_size)
        self.__conv_layer = CNNBlock1D(in_channels=1,
                                       out_channels=1,
                                       kernel_size=kernel_size,
                                       stride=stride)

    @property
    def out_size(self) -> int:
        """
        Returns the length of the of the outputs returned by the module.

        Returns:
            int: length of the outputs returned by the module.
        """
        return self.__out_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Executes a forward pass.

        Args:
            x (Tensor): Nucleotides sequence embeddings 
                        (BATCH SIZE*NB CHUNKS, CHUNK DEPTH, CHUNK LENGTH)

        Returns:
            Tensor: Nucleotides sequences's final embeddings (BATCH SIZE, OUT SIZE)
        """

        # Apply a bottleneck that summarizes all tokens in the sequence
        # (BATCH SIZE*NB CHUNKS, CHUNK DEPTH, CHUNK LENGTH) -> (BATCH SIZE*NB CHUNKS, CHUNK LENGTH)
        x = self.__bottleneck(x).squeeze(dim=1)

        # Reshape the tensor to concatenate chunks associated to each data point
        # (BATCH SIZE*NB CHUNKS, CHUNK LENGTH) -> (BATCH SIZE, 1, CHUNK LENGTH*NB CHUNKS)
        x = x.reshape(-1, 1, self.__chunk_length*self.__nb_chunks)

        # Apply convolution on concatenated chunks
        # (BATCH SIZE, 1, CHUNK LENGTH*NB CHUNKS) -> (BATCH SIZE, OUT SIZE)
        x = self.__conv_layer(x).squeeze(dim=1)

        return x
