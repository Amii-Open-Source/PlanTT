"""
Authors: Nicolas Raymond,
         Ruchika Verma

Description: Stores the class associated to the CNN tower architecture,
             and the class associated to the PNAS CNN model.
             
"""
from torch import bernoulli, cat, nn, ones, Tensor, zeros

# Project imports
from src.data.modules.preprocessing import NUC2ONEHOT_MAPPING
from src.models.blocks import CNNBlock1D, DenseBlock1D
from src.models.plantt import Tower

class CNN1D(Tower):
    """
    1D CNN tower architecture.
    """
    def __init__(self,
                 seq_length: int,
                 feature_dropout: float = 0,
                 dropout: float = 0) -> None:
        """
        Initializes the layers of the models.

        Args:
            seq_length (int): length of the nucleotide sequences expected as inputs.
            feature_dropout (float): Probability of a nucleotide to be replaced by
                                     an 'X' or an 'N' during training. More precisely,
                                     the selected elements are being replaced by the
                                     [0, 0, 0, 0, 1] encoding.
                                     Default to 0.
            dropout (float, optional): probability of an element to be zeroed
                                       in a convolution layer.
                                       Default to 0.
        """
        if seq_length % 8 != 0:
            raise ValueError('seq_length must be divisible by 8')

        super().__init__(int(seq_length/8))

        # Save feature dropout rate
        self.__feature_dropout: float = feature_dropout

        # Build convolution blocks
        self.__conv_blocks = nn.Sequential(CNNBlock1D(in_channels=5,
                                                      out_channels=32,
                                                      kernel_size=3,
                                                      pool_size=None,
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=True,
                                                      add_residual=True),
                                           DenseBlock1D(in_channels=32,
                                                        growth_rate=32,
                                                        nb_conv_block=4,
                                                        dropout=dropout,
                                                        add_residual=False,
                                                        shrink_to_growth_rate=False),
                                           CNNBlock1D(in_channels=32,
                                                      out_channels=64,
                                                      kernel_size=3,
                                                      pool_size=None,
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=True,
                                                      add_residual=True),
                                           DenseBlock1D(in_channels=64,
                                                        growth_rate=64,
                                                        nb_conv_block=4,
                                                        dropout=dropout,
                                                        add_residual=False,
                                                        shrink_to_growth_rate=False),
                                           CNNBlock1D(in_channels=64,
                                                      out_channels=128,
                                                      kernel_size=3,
                                                      pool_size=2,
                                                      pooling_method='max',
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=True,
                                                      add_residual=True),
                                           DenseBlock1D(in_channels=128,
                                                        growth_rate=128,
                                                        nb_conv_block=4,
                                                        dropout=dropout,
                                                        add_residual=False,
                                                        shrink_to_growth_rate=False),
                                           CNNBlock1D(in_channels=128,
                                                      out_channels=64,
                                                      kernel_size=3,
                                                      pool_size=2,
                                                      pooling_method='max',
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=True,
                                                      add_residual=True),
                                           DenseBlock1D(in_channels=64,
                                                        growth_rate=64,
                                                        nb_conv_block=4,
                                                        dropout=dropout,
                                                        add_residual=False,
                                                        shrink_to_growth_rate=False),
                                           CNNBlock1D(in_channels=64,
                                                      out_channels=32,
                                                      kernel_size=3,
                                                      pool_size=2,
                                                      pooling_method='max',
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=True,
                                                      add_residual=True),
                                           DenseBlock1D(in_channels=32,
                                                        growth_rate=32,
                                                        nb_conv_block=4,
                                                        dropout=dropout,
                                                        add_residual=False,
                                                        shrink_to_growth_rate=False),
                                           CNNBlock1D(in_channels=32,
                                                      out_channels=1,
                                                      kernel_size=3,
                                                      pool_size=None,
                                                      stride=1,
                                                      padding=1,
                                                      dropout=dropout,
                                                      extended=False,
                                                      add_residual=False))

    def forward(self, seq: Tensor) -> Tensor:
        """
        Executes a forward pass with the model.
        
        Args:
            seq (tensor): batch of gene sequences (BATCH SIZE, C, SEQ LENGTH)

        Returns:
            tensor: embedding of the gene sequences (BATCH SIZE, SEQ LENGTH/8)
        """
        # If in training, set random nucleotides to [0, 0, 0, 0, 1]
        if self.training and self.__feature_dropout > 0:

            # Generate coordinates to dropout
            bernoulli_samples = bernoulli(ones(seq.shape[0], seq.shape[-1])*self.__feature_dropout)
            coord = bernoulli_samples.nonzero().to(seq.device)

            # Replace the data at these positions
            seq[coord[:, 0], :, coord[:, 1]] = Tensor(NUC2ONEHOT_MAPPING['X']).to(seq.device)

        # Forward pass in the convolution blocks
        # (BATCH SIZE, 5, SEQ LENGTH) -> (BATCH SIZE, SEQ LENGTH/8)
        return self.__conv_blocks(seq).squeeze(dim=1)

class WCNN(nn.Module):
    """
    CNN model based on the architecture proposed in: 
    
    "Evolutionarily informed deep learning methods for predicting
    relative transcript abundance from DNA sequence".
    
    Blocks are separated in two to avoid the following UserWarning:
    
    Using padding='same' with even kernel lengths and odd dilation may require
    a zero-padded copy of the input be created
    
    Padding values provided were chosen to replicate the behavior
    obtained using 'padding=same' with Keras:
    
    See https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/3
    """

    def __init__(self, regression: bool = False) -> None:
        """
        Initializes the layers of the model
        
        regression (bool): if True, no sigmoid activation function will be
                           used at the end of the network.
                           Default to False.
        """
        super().__init__()

        # Block 1
        self.__block_1_0 = nn.Sequential(nn.Conv2d(1, 64,
                                                   kernel_size=(8,5),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_1_1 = nn.Sequential(nn.Conv2d(64, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(2, 0)),
                                         nn.Dropout2d(0.25))

        # Block 2
        self.__block_2_0 = nn.Sequential(nn.Conv2d(64, 128,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_2_1 = nn.Sequential(nn.Conv2d(128, 128,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(1, 0)),
                                         nn.Dropout2d(0.25))

        # Block 3
        self.__block_3_0 = nn.Sequential(nn.Conv2d(128, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_3_1 = nn.Sequential(nn.Conv2d(64, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(1, 0)),
                                         nn.Dropout2d(0.25))
        
        # Linear layers (i.e. fully connected layers)
        self.__linear_layers = nn.Sequential(nn.Flatten(),
                                             nn.Linear(768, 128),
                                             nn.ReLU(),
                                             nn.Dropout1d(0.25),
                                             nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 1))

        # Last activation function
        self.__last_activation = nn.Identity() if regression else nn.Sigmoid()

    def forward(self,
                seq_a: Tensor,
                seq_b: Tensor) -> Tensor:
        """
        Executes a forward pass two batches of DNA sequences.
        
        Args:
            seq_a (tensor): batch of DNA sequences (BATCH SIZE, 5, 3000)
            seq_b (tensor): batch of DNA sequences (BATCH SIZE, 5, 3000)
            
        Returns:
            Tensor: gene expression difference (BATCH SIZE, ). 
        """
        # Reshape the tensors
        # (BATCH SIZE, 5, 3000) -> (BATCH SIZE, 1, 3000, 5)
        seq_a = seq_a.unsqueeze(dim=1).permute(0, 1, 3, 2)
        seq_b = seq_b.unsqueeze(dim=1).permute(0, 1, 3, 2)

        # Concatenate sequences and add 0 padding in the middle
        # (BATCH SIZE, 1, 3000, 5), (BATCH SIZE, 1, 3000, 5) -> (BATCH SIZE, 1, 6030, 5)
        x = cat([seq_a, zeros(seq_a.shape[0], 1, 30, 5).to(seq_a.device), seq_b], dim=2)

        # Forward pass through block 1
        # (BATCH SIZE, 1, 6030, 5) -> (BATCH SIZE, 64, 750, 1)
        x = self.__block_1_1(nn.functional.pad(self.__block_1_0(x), (0, 0, 3, 4)))

        # Forward pass through block 2
        # (BATCH SIZE, 64, 750, 1) -> (BATCH SIZE, 128, 750, 1)
        x = self.__block_2_0(nn.functional.pad(x, (0, 0, 3, 4)))
        # (BATCH SIZE, 128, 750, 1) -> (BATCH SIZE, 128, 94, 1)
        x = self.__block_2_1(nn.functional.pad(x, (0, 0, 3, 4)))

        # Forward pass through block 3
        # (BATCH SIZE, 128, 94, 1) -> (BATCH SIZE, 64, 94, 1)
        x = self.__block_3_0(nn.functional.pad(x, (0, 0, 3, 4)))
        # (BATCH SIZE, 64, 94, 1) -> (BATCH SIZE, 64, 12, 1)
        x = self.__block_3_1(nn.functional.pad(x, (0, 0, 3, 4)))

        # Forward pass in the fully connected layers followed by an activation function
        if self.training:
            return self.__linear_layers(x).squeeze(dim=-1)

        return self.__last_activation(self.__linear_layers(x)).squeeze()
