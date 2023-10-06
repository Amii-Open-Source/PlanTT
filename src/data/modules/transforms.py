"""
Authors: Nicolas Raymond

Description: Stores the scaling method that can be applied to regression targets.
"""
from abc import ABC, abstractmethod
from torch import Tensor


class Scaler(ABC):
    """
    Scaler abstract class
    """
    @abstractmethod
    def __call__(self, y: Tensor) -> Tensor:
        """
        Applies the scaling.

        Args:
            y (Tensor): regression targets (N,)

        Returns:
            Tensor: scaled regression targets (N,)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_inverse_transform(self, y: Tensor) -> Tensor:
        """
        Applies the reverse transformation.
        
        Args:
            y (Tensor): scaled regression target predictions (N,)

        Returns:
            Tensor: regression target predictions in the original scale (N,)
        """
        raise NotImplementedError


class MinMaxScaler(Scaler):
    """
    Scales target values in the range (0, 1) using the following formula:
    
    y' = (y - min)/(max - min)
    """
    def __init__(self,
                 min: float = None,
                 max: float = None) -> None:
        """
        Saves the min and the range values used for the scaling.

        Args:
            min (float): minimum value in the training targets. 
                         If none are provided, the value will be determined during the first call.
                         Default to None.
                         
            max (float): maximum value in the training targets.
                         If none are provided, the value will be determined during the first call.
                         Default to None.
        """
        super().__init__()
        
        if min is not None and max is not None:
            self.__min: float = min
            self.__range: float = max - min
        else:
            self.__min: float = None
            self.__range: float = None
            
    def __call__(self, y: Tensor) -> Tensor:
        """
        Applies the scaling.

        Args:
            y (Tensor): regression targets (N,)

        Returns:
            Tensor: scaled regression targets (N,)
        """
        if self.__range is None:
            self.__min = y.min().item()
            self.__range = y.max().item() - self.__min
            
        return (y - self.__min)/(self.__range)
    
    def apply_inverse_transform(self, y: Tensor) -> Tensor:
        """
        Applies the reverse transformation.
        
        Args:
            y (Tensor): regression target predictions (N,)

        Returns:
            Tensor: regression target predictions in the original scale (N,)
        """
        if self.__range is None:
            raise Exception('Min and max values are still undefined. \
                             The object need to be initialized with the given parameters or called once with a tensor containing target values.')
            
        return y*(self.__range) + self.__min
    
class StandardScaler(Scaler):
    """
    Scales target values in the range (-1, 1) using the following formula:
    
    y' = (y - mean)/std
    """
    def __init__(self,
                 mean: float = None,
                 std: float = None) -> None:
        """
        Saves the mean and the standard deviation.

        Args:
            mean (float, optional): mean of the training targets. Defaults to None.
            std (float, optional): standard deviation of the training targets. Defaults to None.
        """
        super().__init__()
        
        if mean is not None and std is not None:
            self.__mean: float = mean
            self.__std: float = std
        else:
            self.__mean: float = None
            self.__std: float = None
            
    @property
    def mean(self) -> float:
        return self.__mean
    
    @property
    def std(self) -> float:
        return self.__std
            
    def __call__(self, y: Tensor) -> Tensor:
        """
        Applies the transformation.

        Args:
            y (Tensor): regression targets (N,)

        Returns:
            Tensor: scaled regression targets (N,)
        """
        if self.__mean is None:
            self.__mean = y.mean().item()
            self.__std = y.std().item()
            
        return (y - self.__mean)/self.__std
    
    def apply_inverse_transform(self, y: Tensor) -> Tensor:
        """
        Applies the reverse transformation.
        
        Args:
            y (Tensor): regression target predictions (N,)

        Returns:
            Tensor: regression target predictions in the original scale (N,)
        """
        if self.__mean is None:
            raise Exception('Mean and std values are still undefined. \
                             The object need to be initialized with the given parameters or called once with a tensor containing target values.')
            
        return y*self.__std + self.__mean
    