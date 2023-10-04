"""
Authors: Nicolas Raymond

Description: Stores a selection of architecture to use as PlanTT's head.
"""

from src.models.blocks import LinearBlock
from src.models.plantt import Head
from torch import sum as th_sum
from torch import Tensor
from torch.nn import Linear, Sequential


class MLPHead(Head):
    """
    PlanTT head composed of two linear layers separated by an activation function.
    """
    def __init__(self,
                 regression: bool,
                 input_size: int,
                 dropout: float = 0,
                 odd: bool = True) -> None:
        """
        Initializes the layers of the head.
        
        Args:
            regression (bool): if True, a sigmoid activation will be added after the aggregation function.
            input_size (int): length of the embeddings provided as input.
            dropout (float, optional): dropout probabilities in linear blocks. Default to 0.
            odd (bool, optional): if True, the mlp will be an odd function. Default to True.
        """
        super().__init__(regression)
        
        if odd:
            self.__layers = Sequential(LinearBlock(in_features=input_size,
                                                   out_features=50,
                                                   dropout=dropout,
                                                   bias=False,
                                                   use_odd_activation=True),
                                       Linear(in_features=50, out_features=1, bias=False))
        else:
            self.__layers = Sequential(LinearBlock(in_features=input_size,
                                                   out_features=50,
                                                   dropout=dropout,
                                                   bias=True,
                                                   use_odd_activation=False),
                                       Linear(in_features=50, out_features=1, bias=True))
        
    def _aggregate(self,
                   emb_a: Tensor,
                   emb_b: Tensor) -> Tensor:
        """
        Executes a forward pass.

        Args:
            emb_a (Tensor): batch of embeddings (BATCH SIZE, EMB SIZE)
            emb_b (Tensor): batch of embeddings (BATCH SIZE, EMB SIZE)

        Returns:
            Tensor: predictions of gene expression difference (BATCH SIZE, 1)
        """
        return self.__layers(emb_a - emb_b)


class SumHead(Head):
    """
    PlanTT head that simply sums the elements in the vector storing the difference of the embeddings.
    This head is equivariant to input order by construction.
    """
    def __init__(self, regression: bool) -> None:
        """
        Calls the parent constructor to set the activation layer.
        
        Args:
            regression (bool): if True, a sigmoid activation will be added after that aggregation function.
        """
        super().__init__(regression)
        
    def _aggregate(self,
                   emb_a: Tensor,
                   emb_b: Tensor) -> Tensor:
        """
        Sums the elements in the vectors storing the differences of the embeddings.
        
        Args:
            emb_a (Tensor): batch of embeddings (BATCH SIZE, EMB SIZE)
            emb_b (Tensor): batch of embeddings (BATCH SIZE, EMB SIZE)

        Returns:
            Tensor: predictions of gene expression difference (BATCH SIZE, 1)
        """
        return th_sum(emb_a - emb_b, dim=-1)
    
    
    

