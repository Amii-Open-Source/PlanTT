"""
Authors: Nicolas Raymond

Description: Defines the PlanTT model and its abstract blocks.           
"""
from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn import Identity, Module, Sigmoid


class Tower(ABC, Module):
    """
    Abstract class representing a PlanTT tower.
    The tower is in charge of creating embeddings with a single dimension.
    """
    def __init__(self, out_size: int) -> None:
        """
        Calls Module constructor and sets out_size private attribute.
        """
        Module.__init__(self)
        self.__out_size: int = out_size

    @property
    def out_size(self) -> int:
        """
        Returns the number of dimension in the embeddings returned by the tower.

        Returns:
            int: number of dimension in the embeddings returned by the tower.
        """
        return self.__out_size


class Head(ABC, Module):
    """
    Abstract class representing the head of a PlanTT model.
    
    The head is in charge of predicting gene expression difference between two sequences
    based on their embeddings generated by the tower.
    """
    def __init__(self, regression: bool) -> None:
        """
        Calls Module constructor and sets last_activation private attribute.
        
        Args:
            regression (bool): if True, a sigmoid activation will be added after
                               that aggregation function.
        """
        Module.__init__(self)
        self.__last_activation = Identity() if regression else Sigmoid()

    @abstractmethod
    def _aggregate(self,
                   emb_a: Tensor,
                   emb_b: Tensor) -> Tensor:
        """
        Compute predictions of gene expression difference based on the given embeddings.
        
        Args:
            emb_a (Tensor): batch of gene sequence embeddings (BATCH SIZE, EMB SIZE)
            emb_b (Tensor): batch of gene sequence embeddings (BATCH SIZE, EMB SIZE)

        Returns:
            Tensor: predictions of gene expression difference (BATCH SIZE, 1)
        """
        raise NotImplementedError

    def forward(self,
                emb_a: Tensor,
                emb_b: Tensor) -> Tensor:
        """
        Executes a forward pass.

        Args:
            emb_a (Tensor): batch of gene sequence embeddings (BATCH SIZE, EMB SIZE)
            emb_b (Tensor): batch of gene sequence embeddings (BATCH SIZE, EMB SIZE)

        Returns:
            Tensor: predictions of gene expression difference (BATCH SIZE, 1)
        """
        if self.training:
            return self._aggregate(emb_a, emb_b)

        return self.__last_activation(self._aggregate(emb_a, emb_b))

class PlanTT(Module):
    """
    Two-tower model for the prediction of gene expression difference in plants.
    """
    def __init__(self,
                 tower: Tower,
                 head: Head) -> None:
        """
        Sets the underlying tower architecture used to generate the embeddings of the inputs.
        
        Args:
            tower (Tower): model architecture that define the two towers of the model.
            head (Head): model architecture that define the head of the model.
        """
        super().__init__()
        self.__tower: Tower = tower
        self.__head: Head = head

    @property
    def tower(self) -> Tower:
        """
        Returns the Tower.

        Returns:
            Tower: tower of the model.
        """
        return self.__tower

    @property
    def head(self) -> Head:
        """
        Returns the Head.

        Returns:
            Head: head of the model.
        """
        return self.__head

    def forward(self,
                seq_a: Tensor,
                seq_b: Tensor) -> Tensor:
        """
        Executes a forward pass with both batches of sequences. 
        
        Args:
            seq_a (Tensor): batch of DNA sequences (BATCH SIZE, *).
            seq_b (Tensor): batch of DNA sequences (BATCH SIZE, *).

        Returns:
           Tensor: predictions of gene expression difference (BATCH_SIZE, )
        """

        # Generate the embeddings with the tower
        # (BATCH SIZE, *) -> (BATCH SIZE, EMB_SIZE)
        emb_a, emb_b = self.__tower(seq_a), self.__tower(seq_b) 

        # Pass the embeddings to the head
        return self.__head(emb_a, emb_b).squeeze(dim=-1)
