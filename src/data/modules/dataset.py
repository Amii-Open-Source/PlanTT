"""
Authors: Nicolas Raymond
         Fatima Davelouis
         Ruchika Verma
         
Description: Stores the class the dataset.
"""

from json import load as jsload
from os.path import join
from pickle import load
from torch import abs, bernoulli, cat, flip, LongTensor, nn, no_grad, Tensor, vstack, zeros
from torch.utils.data import Dataset
from typing import Callable
from warnings import warn


# Project imports
from settings.paths import REVERSE_COMPLEMENT_MAP
from src.data.modules.transforms import Scaler

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
        1) If include_flip is True and include_reverse_complement is False,
           the size of the dataset is doubled.
        2) If include_flip is False and include_reverse_complement is True,
           the size of the dataset is doubled.
        3) If include_flip is True and include_reverse_complement is True,
           the size of the dataset is quadrupled.
        
        Args:
            seq_a (Tensor): encoded gene sequences (N, C, 3000) or 
                            tokenized gene sequences (N, NB CHUNKS, 512).
            seq_b (Tensor): encoded gene sequences (N, C, 3000) or 
                            tokenized gene sequences (N, NB CHUNKS, 512).
            labels (Tensor): binary labels indicating expression difference between
                             pairs of gene sequences.
            species (Tensor): ids associated to the specie pair of each observation.
            regression (bool, optional): if True, indicates that labels are associated to a 
                                         regression task. Otherwise the object assumes that
                                         labels are associated to a binary classification task. 
                                         Default to False.
            include_flip (bool, optional): if True, flipped versions of the pairs of 
                                           genes are also included. 
                                           Default to False.
            include_reverse_complement (bool, optional): if True, reverse complement of the pairs 
                                                         of genes are also included. 
                                                         Default to False.
            tokens (bool, optional): if True, the sequences provided are expected
                                     to be tokenized sequences. 
                                     Default to False.
            scaler (Scaler, optional): if given, the scaling method is applied to the targets.
                                       Default to None.
        """
        # Set the 'flip' private method
        self.__flip: Callable = self.flip_reg_labels if regression else self.flip_class_labels

        # Set the 'get_reverse_complement' private method
        if tokens:
            self.__get_reverse_complement: Callable = self.get_token_reverse_complement
        else:
            self.__get_reverse_complement: Callable = self.get_one_hot_reverse_complement

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

        self.__seq_a: Tensor = seq_a
        self.__seq_b: Tensor = seq_b
        self.__species: Tensor = species

        # Save the scaler
        if scaler is not None:
            self.scaler: Scaler = scaler
            self.__labels: Tensor = self.scaler(labels)
        else:
            self.scaler: Scaler = None
            self.__labels: Tensor = labels

        # Set the 'get' private method
        self.__get: Callable = self.__eval_getter

        # Save attributes indicating if flipped pairs or reverse complements are included
        self.__flip_included: bool = include_flip
        self.__reverse_complement_included: bool = include_reverse_complement

    @property
    def labels_scale(self) -> float:
        """
        Returns the standard deviation of the labels.

        Returns:
            float: standard deviation of the labels.
        """
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
            regression (bool, optional): if True, regression targets will be used as labels.
                                         Default to False.
            category (int, optional): if given, only the observations matching the
                                      category will be kept.
                                      Default to None.
            specie (int, optional): if given, only the observations matching the
                                    specie id will be kept.
                                    Default to None.                         
            include_flip (bool, optional): if True, flipped versions of the pairs
                                           of genes are also included. 
                                           Default to False.
            include_reverse_complement (bool, optional): if True, reverse complement of the pairs
                                                         of genes are also included. 
                                                         Default to False.
            tokens(bool, optional): if True, indicates that the data 
                                    extracted contain tokenized sequences.
                                    Default to False.
            scaler (Scaler, optional): if given, the scaling method is applied to the targets.
                                       Default to None.
        """
        with open(path, 'rb') as file:
            _, x, y_class, y_reg, categories, species, _ = load(file)

        # Filter data according to given category
        if category is not None:
            if category not in [0, 1, 2, 3]:
                raise ValueError('category must be one of [0, 1, 2, 3]')

            idx_to_keep = (categories <= category).nonzero().squeeze()
            x, y_class = x[idx_to_keep], y_class[idx_to_keep]
            y_reg, species = y_reg[idx_to_keep], species[idx_to_keep]

        # Filter data according to the given specie
        if specie is not None:
            if specie not in cls.PAIRS_TO_SPECIE_ID_MAP.values():
                raise ValueError(f'specie must be one of {cls.PAIRS_TO_SPECIE_ID_MAP.values()}')

            idx_to_keep = (species == specie).nonzero().squeeze()
            x, y_class = x[idx_to_keep], y_class[idx_to_keep]
            y_reg, species = y_reg[idx_to_keep], species[idx_to_keep]

        # If data was filtered
        if category is not None or specie is not None:

            # Balance the remaining data
            x, y_class, y_reg, _ = cls.balance_data(x, y_class, y_reg)

        # If current data holds tokenized sequences (N, 2, NB CHUNKS, 512)
        if tokens:
            seq_a = x[:, 0, :, :].squeeze()
            seq_b = x[:, 1, :, :].squeeze()

        # Otherwise (if data holds encoded sequences with C channels) (N, C, 6030)
        else:
            seq_a = x[:, :, :3000].float()
            seq_b = x[:, :, 3030:].float()

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
            warn('Train mode might not be effective since flipped pairs \
                  of genes are already included in the dataset')
        if self.__reverse_complement_included:
            warn('Train mode might not be effective since reverse complement \
                  of pairs of genes are already included in the dataset')

        self.__get = self.__train_getter

    def eval(self) -> None:
        """
        Sets the dataset into eval mode.
        
        In eval mode, items are extracted as they are. 
        None of them are being flipped or changed for their reverse complement.
        """
        self.__get = self.__eval_getter

    def __train_getter(self, idx: list[int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns items with a probability 0.5 of being flipped and a probability 0.5
        of being changed for their reverse complement.

        Args:
            idx (list[int]): indexes of inputs to get.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: seq_a (N, 5, 3000), seq_b (N, 5, 3000),
                                                   labels (N,), species (N,)
        """
        # Sample a value from a Bernoulli distribution
        change_to_reverse_complement = bool(bernoulli(Tensor([0.5])).item())
        flip_pairs = bool(bernoulli(Tensor([0.5])).item())

        # According to the success (or fail) associated to Bernoulli sample, 
        # return the items in the appropriate format
        if change_to_reverse_complement and flip_pairs:
            seq_a = self.__get_reverse_complement(self.__seq_b[idx])
            seq_b = self.__get_reverse_complement(self.__seq_a[idx])
            labels = self.__flip(self.__labels[idx])
            return seq_a, seq_b, labels, self.__species[idx]

        if change_to_reverse_complement:
            seq_a = self.__get_reverse_complement(self.__seq_a[idx])
            seq_b = self.__get_reverse_complement(self.__seq_b[idx])
            return seq_a, seq_b, self.__labels[idx], self.__species[idx]

        if flip_pairs:
            seq_a = self.__seq_b[idx]
            seq_b = self.__seq_a[idx]
            labels = self.__flip(self.__labels[idx])
            return seq_a, seq_b, labels, self.__species[idx]

        return self.__seq_a[idx], self.__seq_b[idx], self.__labels[idx], self.__species[idx]


    def __eval_getter(self, idx: list[int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns items in their original form.
        
        Args:
            idx (list[int]): indexes of inputs to get.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: seq_a (N, 5, 3000), seq_b (N, 5, 3000),
                                                   labels (N,), species (N,)
        """
        return self.__seq_a[idx], self.__seq_b[idx], self.__labels[idx], self.__species[idx]


    @staticmethod
    def flip_class_labels(labels: Tensor) -> Tensor:
        """
        Changes 0s to 1s and 1s to 0s.
        
        Args:
            labels (Tensor): binary classification labels (N,).

        Returns:
            Tensor: flipped binary classification labels (N,).
        """
        return abs(labels - 1)

    @staticmethod
    def flip_reg_labels(labels: Tensor) -> Tensor:
        """
        Changes the sign of the regression labels.

        Args:
            labels (Tensor): regression labels (N,).

        Returns:
            Tensor: flipped regression labels (N,).
        """
        return -labels

    @staticmethod
    def get_one_hot_reverse_complement(seq: Tensor) -> Tensor:
        """
        Reverses the nucleotide sequences and switches A <-> T and C <-> G.
        
        The encodings have either 5 dimensions
        
         "A": [1, 0, 0, 0, 0],
         "C": [0, 1, 0, 0, 0],
         "G": [0, 0, 1, 0, 0],
         "T": [0, 0, 0, 1, 0],
         "N": [0, 0, 0, 0, 1],
         "X": [0, 0, 0, 0, 1]
         
        or 4 dimensions
        
         "A": [1, 0, 0, 0],
         "C": [0, 1, 0, 0],
         "G": [0, 0, 1, 0],
         "T": [0, 0, 0, 1],
         "N": [.25, .25, .25, .25],
         "X": [.25, .25, .25, .25]
         
        Hence, the function switches idx 0 <-> 3 and 1 <-> 2 in the channel dimension, 
        then flip on the length dimension.

        Args:
            seq (Tensor): gene sequences (N, C, 3000).

        Returns:
            Tensor: reverse complement of gene sequences (N, C, 3000).
        """
        if seq.shape[1] == 5:
            return seq[:, LongTensor([3, 2, 1, 0, 4]), :].flip(dims=(2,))

        return seq[:, LongTensor([3, 2, 1, 0]), :].flip(dims=(2,))

    @staticmethod
    def get_token_reverse_complement(seq: Tensor) -> Tensor:
        """
        Reverses the nucleotide sequences and switches A <-> T and C <-> G.
        
        To do so, the function first replaces each k-mer token id for the token id
        associated to the reverse complement and then flip the order of all token ids.
        
        Args:
            seq (Tensor): tokenized gene sequences (N, NB CHUNKS, 512).

        Returns:
            Tensor: reverse complement of gene sequences (N, NB CHUNKS, 512).
        """
        # Load the reverse complement map
        id_to_reverse_comp_id = OCMDataset.load_reverse_complement_map()

        # We create an embedding layer to swap values without the use of a for loop
        with no_grad():
            embedding_layer = nn.Embedding(len(id_to_reverse_comp_id), 1).requires_grad_(False)
            nn.init.constant_(embedding_layer.weight, 1)
            reverse_comp_ids = LongTensor(list(id_to_reverse_comp_id.values()))
            embedding_layer.weight.multiply_(reverse_comp_ids).view(-1,1)
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
        with open(join(REVERSE_COMPLEMENT_MAP, f'map_{k}'), 'r', encoding="utf-8") as file:
            id_to_reverse_comp_id = jsload(file)

        # Change the type of the keys to int
        return {int(k): v for k,v in id_to_reverse_comp_id.items()}

    @staticmethod
    def balance_data(x: Tensor,
                     y_class: Tensor,
                     y_reg: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Modifies data so the quantity of 0 and 1 labels are approximately the same.

        Args:
            x (Tensor): encoded gene sequences (N, C, 6030) or 
                        tokenized gene sequences (N, 2, NB CHUNKS, 512).
            y_class (Tensor): binary classification labels (N,).
            y_reg (Tensor): regression targets (N,).

        Returns:
            tuple[Tensor, Tensor, Tensor]: updated data (x, y_class, y_reg) and binary mask.
        """
        # Find the indexes where data should be modified
        idx = OCMDataset.find_idx_to_flip(y_class)

        # Create binary mask indicating the location of elements that are flipped
        mask = zeros(x.shape[0])

        # Modify the data
        if len(idx) > 0:

            # Flip concerned binary labels
            y_class[idx] = OCMDataset.flip_class_labels(y_class[idx])

            # Change the sign of concerned regression targets
            y_reg[idx] = OCMDataset.flip_reg_labels(y_reg[idx])

            # Switch the order of concerned sequences
            if x.shape[1] == 2: # If sequences are tokenized
                x[idx] = flip(x[idx], dims=[1])
            else:
                x[idx] = cat([x[idx, :, 3030:], x[idx, :, 3000:3030], x[idx, :, :3000]], dim=-1)

            # Update the mask
            mask[idx] = 1

        return x, y_class, y_reg, mask

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
