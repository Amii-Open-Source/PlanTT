"""
Authors: Nicolas Raymond
         Fatima Davelouis
         Ruchika Verma
         
Description: Stores the classes defining the different type of datasets used during the experiment.
"""

from json import load as jsload
from os.path import join
from pickle import load
from settings.paths import REVERSE_COMPLEMENT_MAP
from torch import abs, bernoulli, cat, flip, LongTensor, nn, no_grad, ones, Tensor, vstack, where, zeros
from src.data.modules.transforms import Scaler
from torch.utils.data import Dataset
from typing import Callable
from warnings import warn


class OCMDataset(Dataset):
    """
    Datasets dedicated to orthologs contrast models such as PlanTT.
    """
    SEQ_LENGTH: int = 3000
    PAIRS_TO_SPECIE_ID_MAP: dict = {('medicago', 'faba'): 0,
                                    ('pea', 'faba'): 10,
                                    ('pea', 'medicago'): 20,
                                    ('sorghum',  'mays'): 30}
    
    def __init__(self,
                 seq_a: Tensor,
                 seq_b: Tensor,
                 labels: Tensor,
                 species: Tensor,
                 regression: bool = False,
                 include_flip: bool = False,
                 include_reverse_complement: bool = False,
                 tokens: bool = False,
                 scaler: Scaler = None) -> None:
        """
        Saves the features, the labels, and three private methods based on the arguments given.
        
        Note:
        1) If include_flip is True and include_reverse_complement is False, the size of the dataset is doubled.
        2) If include_flip is False and include_reverse_complement is True, the size of the dataset is doubled.
        3) If include_flip is True and include_reverse_complement is True, the size of the dataset is quadrupled.
        
        Args:
            seq_a (Tensor): gene sequences (N, 1, 3000, 5) or tokenized gene sequences (N, NB_CHUNKS, 512).
            seq_b (Tensor): gene sequences (N, 1, 3000, 5) or tokenized gene sequences (N, NB_CHUNKS, 512).
            labels (Tensor): binary labels associated to gene expressivity of pairwise gene sequence.
            species (Tensor): ids associated to the specie pair of each observation.
            regression (bool, optional): if True, indicates that labels are associated to a regression task.
                                         Otherwise the object assumes that labels are associated to a binary classification task. Default to False.
            include_flip (bool, optional): if True, flipped versions of the orthologs pairs are also included. Default to False.
            include_reverse_complement (bool, optional): if True, reverse complement of the orthologs pairs are also included. Default to False.
            tokens (bool, optional): if True, the sequences provided are expected to be tokenized sequences. Default to False.
            scaler (Callable, optional): if given, the scaling method is applied to the targets.
        """
        # Set the 'flip' private method
        self.__flip: Callable[[Tensor], Tensor] = self.flip_reg_labels if regression else self.flip_class_labels
        
        # Set the 'get_reverse_complement' private method
        self.__get_reverse_complement: Callable[[Tensor], Tensor] = self.get_token_reverse_complement if tokens else self.get_one_hot_reverse_complement
        
        # Save the data
        if include_reverse_complement:
            seq_a = vstack([seq_a, self.__get_reverse_complement(seq_a)])
            seq_b = vstack([seq_b, self.__get_reverse_complement(seq_b)])
            labels = cat([labels, labels])
            species = cat([species, species])
            
        if include_flip:
            temp_seq = vstack([seq_a, seq_b])
            seq_b = vstack([seq_b, seq_a])
            seq_a  = temp_seq
            labels = cat([labels, self.__flip(labels)])
            species = cat([species, species])
        
        self.__seq_a = seq_a
        self.__seq_b = seq_b
        self.__species = species
        
        # Save the scaler
        if scaler is not None:
            self.scaler: Callable[[Tensor], Tensor] = scaler
            self.__labels: Tensor = self.scaler(labels)
        else:
            self.scaler = None
            self.__labels: Tensor = labels
     
        # Set the 'get' private method
        self.__get: Callable[[list[int]], tuple[Tensor, Tensor, Tensor]] = self.__eval_getter
        
        # Save an attribute indicating if reverse were included
        self.__flip_included: bool = include_flip
        self.__reverse_complement_included: bool = include_reverse_complement
        
    @property
    def labels_scale(self) -> float:
        return self.__labels.std()
  
    @classmethod
    def from_path(cls,
                  path: str,
                  regression: bool = False,
                  category: int = None,
                  specie: int = None,
                  include_flip: bool = True,
                  include_reverse_complement: bool = False,
                  tokens: bool = False,
                  scaler: Scaler = None):
        """
        Creates an instance from the data contained in a pickle file.

        Args:
            path (str): path of the pickle file.
            regression (bool, optional): if True, regression targets will be used as labels. Defaults to False.
            category (int, optional): if given, only the observations matching the category will be kept. Defaults to None.
            specie (int, optional): if given, only the observations matching the specie id will be kept. Defaults to None.                         
            include_flip (bool, optional): if True, flipped versions of the orthologs pairs are also included. Default to False.
            include_reverse_complement (bool, optional): if True, reverse complement of the orthologs pairs are also included. Default to False.
            tokens(bool, optional): if True, indicates that the data extracted contain tokenized sequences.
            scaler (Callable, optional): if given, the scaling method is applied to the targets.
        """
        with open(path, 'rb') as file:
                _, x, y_class, y_reg, categories, species = load(file)
                
        # Filter data according to given category
        if category is not None:
            if category not in [0, 1, 2, 3]:
                raise ValueError('Category must be one of [0, 1, 2, 3]')
            
            idx_to_keep = (categories <= category).nonzero().squeeze()
            x, y_class, y_reg, species = x[idx_to_keep], y_class[idx_to_keep], y_reg[idx_to_keep], species[idx_to_keep]
            
        # Filter data according to given specie
        if specie is not None:
            if specie not in cls.PAIRS_TO_SPECIE_ID_MAP.values():
                raise ValueError(f'Specie must be one of {cls.PAIRS_TO_SPECIE_ID_MAP.values()}')
            
            idx_to_keep = (species == specie).nonzero().squeeze()
            x, y_class, y_reg, species = x[idx_to_keep], y_class[idx_to_keep], y_reg[idx_to_keep], species[idx_to_keep]
        
        # If data was filtered
        if category is not None or specie is not None:
            
            # Balance the remaining data
            x, y_class, y_reg = cls.balance_data(x, y_class, y_reg)
        
        # If current data holds tokenized sequences (N, 2, NB_CHUNKS, 512)
        if tokens:
            seq_a = x[:, 0, :, :].squeeze()
            seq_b = x[:, 1, :, :].squeeze()
        
        # Otherwise (if data holds one-hot encoded sequences) (N, 1, 6030, 5)
        else:
            seq_a = x[:, :, :3000, :].float()
            seq_b = x[:, :, 3030:, :].float()
            
        labels = y_reg.float() if regression else y_class.float()
        
        return cls(seq_a=seq_a,
                   seq_b=seq_b,
                   labels=labels,
                   species=species,
                   regression=regression,
                   include_flip=include_flip,
                   include_reverse_complement=include_reverse_complement,
                   tokens=tokens,
                   scaler=scaler)

    def train(self) -> None:
        """
        Sets the dataset into training mode.
        In training mode, items have a probability 0.5 of being flipped or 
        changed for their reverse complement when extracted from the dataset.
        """
        if self.__flip_included:
            warn('Train mode might not be effective since flipped orthologs pairs are already included in the dataset')
        if self.__reverse_complement_included:
            warn('Train mode might not be effective since reverse complement of orthologs pairs are already included in the dataset')
            
        self.__get = self.__train_getter
        
    def eval(self) -> None:
        """
        Sets the dataset into eval mode.
        In eval mode, items are extracted as they are. None are being flipped or changed for their reverse complement.
        """
        self.__get = self.__eval_getter
        
    def __train_getter(self, idx: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Get items with a probability 0.5 of being flipped and a probability 0.5
        of being changed for their reverse complement.

        Args:
            idx (list[int]): indexes of inputs to get.

        Returns:
            tuple[Tensor, Tensor, Tensor]: seq_a, seq_b, labels (BATCH_SIZE, 1, 3000, 5)
        """
        # Sample a value from a Bernoulli distribution
        change_to_reverse_complement = bool(bernoulli(Tensor([0.5])).item())
        flip = bool(bernoulli(Tensor([0.5])).item())
        
        # According to the success (or fail) associated to Bernoulli sample, return the items in the appropriate format
        if change_to_reverse_complement and flip:
            return self.__get_reverse_complement(self.__seq_b[idx]), self.__get_reverse_complement(self.__seq_a[idx]), self.__flip(self.__labels[idx]), self.__species[idx]
        
        elif change_to_reverse_complement:
            return self.__get_reverse_complement(self.__seq_a[idx]), self.__get_reverse_complement(self.__seq_b[idx]), self.__labels[idx], self.__species[idx]
        
        elif flip:
            return self.__seq_b[idx], self.__seq_a[idx], self.__flip(self.__labels[idx]), self.__species[idx]
        
        else:
            return self.__seq_a[idx], self.__seq_b[idx], self.__labels[idx], self.__species[idx]
        
        
    def __eval_getter(self, idx: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        """
        Get items in their original form.
        
        Args:
            idx (list[int]): indexes of inputs to get.

        Returns:
            tuple[Tensor, Tensor, Tensor]: seq_a, seq_b, labels (BATCH_SIZE, 1, 3000, 5)
        """
        return self.__seq_a[idx], self.__seq_b[idx], self.__labels[idx], self.__species[idx]
    
    
    @staticmethod
    def flip_class_labels(labels: Tensor) -> Tensor:
        """
        Changes 0s to 1s and 1s to 0s.
        
        Args:
            labels (Tensor): binary classification labels.

        Returns:
            Tensor: flipped binary classification labels.
        """
        return abs(labels - 1)
    
    @staticmethod
    def flip_reg_labels(labels: Tensor) -> Tensor:
        """
        Changes the sign of the regression labels.

        Args:
            labels (Tensor): regression labels (i.e. targets).

        Returns:
            Tensor: flipped regression labels.
        """
        return -labels
    
    @staticmethod
    def get_one_hot_reverse_complement(seq: Tensor) -> Tensor:
        """
        Reverses the nucleotide sequences and switches A <-> T and C <-> G.
        
        The one-hot encodings are:
        
         "A": [1, 0, 0, 0, 0],
         "C": [0, 1, 0, 0, 0],
         "G": [0, 0, 1, 0, 0],
         "T": [0, 0, 0, 1, 0],
         "N": [0, 0, 0, 0, 1],
         "X": [0, 0, 0, 0, 1]
         
        Hence, the function switches idx 0 <-> 3 and 1 <-> 2 in the last dimension, 
        then flip on the dimension 2.

        Args:
            seq (Tensor): gene sequences (N x 1 x 3000 x 5).

        Returns:
            Tensor: reverse complement of gene sequences (N, 1, 3000, 5).
        """
        return seq[:, :, :, LongTensor([3, 2, 1, 0, 4])].flip(dims=(2,))
    
    @staticmethod
    def get_token_reverse_complement(seq: Tensor) -> Tensor:
        """
        Reverses the nucleotide sequences and switches A <-> T and C <-> G.
        
        To do so, the function first replaces each k-mer token id for the token id
        associated to the reverse complement and then flip the order of all token ids.
        
        Args:
            seq (Tensor): tokenized gene sequences (N, NB_CHUNKS, 512).

        Returns:
            Tensor: reverse complement of gene sequences (N, NB_CHUNKS, 512).
        """
        # Load the reverse complement map
        id_to_reverse_comp_id = OCMDataset.load_reverse_complement_map()
        
        # We create an embedding layer to swap values without the use of a for loop
        with no_grad():
            embedding_layer = nn.Embedding(len(id_to_reverse_comp_id), 1).requires_grad_(False)
            nn.init.constant_(embedding_layer.weight, 1)
            embedding_layer.weight.multiply_(LongTensor(list(id_to_reverse_comp_id.values())).view(-1,1))
            reverse_complement = embedding_layer(seq).squeeze(dim=-1)
            
        return reverse_complement.flip(dims=(2,))
    
    @staticmethod
    def load_reverse_complement_map(k: int = 6) -> dict[int, int]:
        """
        Loads a dictionary mapping each k-mer token id to the id of the
        token associated to its reverse complement.
        
        Args:
            k (int, optional): map identifier. Defaults to 6.

        Returns:
            dict[int, int]: id to reverse complement id map.
        """
        # Load the dictionary
        with open(join(REVERSE_COMPLEMENT_MAP, f'map_{k}'), 'r') as file:
            id_to_reverse_comp_id = jsload(file)
            
        # Change the type of the keys to int
        return {int(k): v for k,v in id_to_reverse_comp_id.items()}
        
    @staticmethod
    def balance_data(x: Tensor,
                     y_class: Tensor,
                     y_reg: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Modifies data so the quantity of 0 and 1 labels are approximately the same.

        Args:
            x (Tensor): encoded gene sequences (N, 1, 6030, 5) or tokenized gene sequences (N, 2, NB_CHUNKS, 512)
            y_class (Tensor): binary classification labels.
            y_reg (Tensor): regression targets (rank differences).

        Returns:
            tuple[Tensor, Tensor, Tensor]: updated data (x, y_class, y_reg).
        """
    
        # Find the indexes where data should be modified
        idx = OCMDataset.find_idx_to_flip(y_class)
        
        # Modify the data
        if len(idx) > 0:
            
            # Flip concerned binary labels
            y_class[idx] =  OCMDataset.flip_class_labels(y_class[idx])
            
            # Change the sign of concerned regression targets
            y_reg[idx] = OCMDataset.flip_reg_labels(y_reg[idx])
            
            # Switch the order of concerned sequences
            if x.shape[1] == 2: # If sequences are tokenized
                x[idx] = flip(x[idx], dims=[1])
            else:
                x[idx] = cat([x[idx, :, 3000:, :], x[idx, :, :3000, :]], dim=-1)
            
        return x, y_class, y_reg
    
    @staticmethod
    def find_idx_to_flip(binary_class_labels: Tensor) -> Tensor:
        """
        Find indexes where binary class labels should be changed to ensure
        each class labels represent approximately 50% of the total.
        
        Args:
            binary_class_labels (Tensor): tensor with binary values

        Returns:
            Tensor: tensor with idx of label to change
        """
        # Approximate the number of elements expected in each class 
        expected_total =  binary_class_labels.shape[0]//2
        
        # Count the number of elements in each class
        zeros_count = (binary_class_labels == 0).sum().item()
        ones_count = binary_class_labels.shape[0] - zeros_count
        
        # Return an empty tensor if the number of elements in each class is already equal
        if zeros_count == ones_count:
            return Tensor([])

        # Save the less represented class
        if zeros_count < ones_count:
            minority, minority_count = 0, zeros_count
        else:
            minority, minority_count = 1, ones_count
            
        # Calculate the number of elements in the minority class that should be changed
        qty_to_flip = expected_total - minority_count
        
        # Identify the idx of the elements to change (looking from left to right)
        idx_to_flip = []
        for i, label in enumerate(binary_class_labels.tolist()):
            
            # Stop the for loop if the number of elements is reached
            if len(idx_to_flip) == qty_to_flip:
                break
            
            # Add idx to the list
            if label != minority:
                idx_to_flip.append(i)
            
        return Tensor(idx_to_flip).long()
        
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        return self.__get(idx)
    
    def __len__(self) -> int:
        return self.__labels.shape[0]
    
    
class MLMDataset(Dataset):
    """
    Datasets dedicated to Masked Language Models (MLMs)
    """
    def __init__(self,
                 x: Tensor,
                 mask: Tensor) -> None:
        """
        Stores the tensor containing tokenized DNA sequences.

        Args:
            x (Tensor): tokenized DNA sequences (N, 512).
            mask (Tensor): binary mask associated to tokenized DNA sequences for masked language pre-training (N, 512).
        """
        super().__init__()
        self.__x = x
        self.__mask = mask
        
    @classmethod
    def from_path(cls, path: str):
        """
        Creates an instance from the data contained in a pickle file.

        Args:
            path (str): path of the pickle file.
        """
        with open(path, 'rb') as file:
            x, mask = load(file)
            
        return cls(x, mask)

    def __getitem__(self, idx) -> Tensor:
        return self.__x[idx], self.__mask[idx]
    
    def __len__(self) -> int:
        return self.__x.shape[0]
    
    
class EncoderDecoderDataset(Dataset):
    """
    Dataset dedicated to the pre-training of Encoder-Decoder model.
    """
    def __init__(self,
                 x: Tensor,
                 masking_percentage: float = 0,
                 masking_span: int = 3) -> None: 
        """
        Stores the tensor containing the one-hot encoded sequences.
        
        Args:
            x (Tensor): one-hot encoded sequences (N, 1, 3000, 5)
            masking_percentage (float): Percentage of elements replaced by an 'X' or an 'N' during masking.
                                        More precisely, the selected elements are being replaced by the [0, 0, 0, 0, 1] encoding.
            masking_span (int): number of contiguous nucleotides masked within each window.
        """
        self.__x = x
        if masking_percentage > 0:
            self.__mask = self.create_mask(self.__x.shape[0], self.__x.shape[2], masking_percentage/masking_span, masking_span)
            self.__get: Callable = self.__default_getter
        else:
            self.__mask = None
            self.__get: Callable = self.__no_mask_getter
        
    @classmethod
    def from_path(cls,
                  path: str,
                  masking_percentage: float = 0):
        """
        Creates an instance from the data contained in a pickle file.

        Args:
            path (str): path of the pickle file.
            masking_percentage (float): Feature dropout probability used during the creation of the mask.
                                     It represents the probability of a nucleotide to be replaced by an 'X' or an 'N'.
                                     More precisely, the selected elements are being replaced by the [0, 0, 0, 0, 1] encoding.
        """
        with open(path, 'rb') as file:
                _, x, _, _, _, _ = load(file)
        
        x = cat([x[:, :, :3000, :], x[:, :, 3030:, :]]).float() # (N, 1, 6030, 5) -> (2N, 1, 3000, 5)
                
        return cls(x=x, masking_percentage=masking_percentage)
    
    @staticmethod
    def create_mask(nb_sequences: int,
                    sequence_length: int,
                    span_start_proba: float,
                    masking_span: int) -> Tensor:
        """
        Generates a mask where coordinates of the basepairs that need to be masked are set to 1 and others to 0.

        Args:
            nb_sequences (int): number of sequences.
            sequence_length (int): length of each sequence.
            masking_percentage (float): percentage of elements to mask.
            span_start_proba (float): probability of a nucleotide to be selected as a starting point for span masking.
            masking_span (int): number of contiguous nucleotides masked within each window.

        Returns:
            Tensor: coordinates of elements to mask (*, 2)
        """
        # Sample mask starting points
        mask = bernoulli(ones(nb_sequences, sequence_length)*span_start_proba).long()
        
        # Extend mask to the masking span
        shift = mask
        for _ in range(masking_span):
            shift = cat((zeros((shift.shape[0], 1)).long(), shift[:, 0:(shift.shape[1]-1)]), 1)
            mask += shift
            
        # Modify mask to make sure all values are binary (if two mask spans overlapped there might be values > 1)
        return where(mask > 1, 1, mask)
    
    def __default_getter(self, idx) -> tuple[Tensor, Tensor]:
        return self.__x[idx], self.__mask[idx]
    
    def __no_mask_getter(self, idx) -> Tensor:
        return self.__x[idx]
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor] | Tensor:
        return self.__get(idx)
    
    def __len__(self) -> int:
        return self.__x.shape[0]
    
    