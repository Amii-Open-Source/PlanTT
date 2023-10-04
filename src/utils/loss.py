"""
Authors: Nicolas Raymond

Description: Custom loss functions.
"""

from abc import ABC, abstractmethod
from torch import abs, clamp, log, pow, Tensor
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

class Loss(ABC):
    """
    Loss abstract class.
    """
    def __init__(self,
                 name: str,
                 regression: bool) -> None:
        """
        Sets the name and the type of the loss function.

        Args:
            name (str): name of the loss function.
            regression (bool): if True, indicates that the loss is used for regression.
                               Otherwise, the loss is used for classification.
        """
        self.__name = name
        self.__regression = regression
        
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def is_for_regression(self) -> bool:
        return self.__regression
        
    @abstractmethod
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the loss based on the predicted values and the targets.
        
        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """
        raise NotImplementedError
    
class BinaryCrossEntropy(Loss):
    """
    Binary cross entropy loss function.
    """
    def __init__(self) -> None:
        """
        Initializes the loss.
        """
        super().__init__(name='BCE', regression=False)
        
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the loss based on the predicted values and the targets.
        
        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """
        return binary_cross_entropy_with_logits(predicted_values, targets)
    
    
class MSE(Loss):
    """
    Mean squared error loss.
    
    Each term in the sum is represented as follow:
    (y_hat - y)^2
    """
    def __init__(self) -> None:
        """
        Sets the name of the loss function.
        """
        super().__init__(name='MSE', regression=True)
        
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the mean squared error.

        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """

        return mse_loss(predicted_values, targets)
    

class MeanAbsolutePercentageError(Loss):
    """
    Mean Absolute Percentage Error (MAPE) loss function.
    
    Each term in the sum is represented as follow:
    |y_hat - y|/y
    """
    def __init__(self, epsilon: float = 1.17e-06) -> None:
        """
        Sets the epsilon value.
        
        Args:
            epsilon (float, optional): Specifies the lower bound for target values. Any target value below epsilon
                                       is set to epsilon (avoids ``ZeroDivisionError``). Defaults to 1.17e-06.
        """
        super().__init__(name='MAPE-loss', regression=True)
        self.__epsilon: float = epsilon
    
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the mean absolute percentage error (MAPE).

        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """

        return (abs(predicted_values - targets) / clamp(abs(targets), min=self.__epsilon)).mean()
    

class WeightedMSE(Loss):
    """
    Weighted mean squared error loss.
    
    Each term in the sum is represented as follow:
    ((y_hat - y)^2)/y^2
    """
    def __init__(self) -> None:
        """
        Sets the name of the loss function.
        """
        super().__init__(name='wMSE', regression=True)
        
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the weighted mean squared error.

        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """
        individual_losses = mse_loss(predicted_values, targets, reduction='none')

        return (individual_losses/pow(targets, 2)).mean()
      
class MultitaskLoss(Loss):
    """
    Loss that constitutes a weighted average of the mean squared error and the binary cross entropy.
    """
    def __init__(self,
                 sigma: float,
                 beta: float = 0.5) -> None:
        """
        Sets the name of the metric and saves the alpha parameter

        Args:
            sigma (float): loss scaling factor (std of targets in training set)
            beta (float): coefficient multiply mean squared error loss
        """
        super().__init__(name='MSE-BCE', regression=True)
        self.__sigma: float = sigma
        self.__beta: float = beta
        
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the weighted average of the mean squared error and the BCE losses.

        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.

        Returns:
            Tensor: loss (1,)
        """
        regression_loss = mse_loss(predicted_values, targets)/(self.__sigma**2)
        classification_loss = binary_cross_entropy_with_logits(predicted_values/self.__sigma, (targets > 0).float())

        return self.__beta*regression_loss + (1-self.__beta)*classification_loss
    
class UncertaintyMultitaskLoss(Loss):
    """
    Loss combining the mean squared error and the binary cross entropy using uncertainty parameters.
    
    The weight assigned to each loss is learned. The implementation is based on the paper:
    https://arxiv.org/pdf/1705.07115.pdf
    """
    def __init__(self) -> None:
        """
        Sets the name of the metric.
        """
        super().__init__(name='u-MSE-BCE', regression=True)
        
    def __call__(self,
                predicted_values: Tensor,
                targets: Tensor,
                sigma_1: Tensor,
                sigma_2: Tensor) -> Tensor:
        """
        Calculates the weighted mean squared error.

        Args:
            predicted_values (Tensor): values predicted by a model.
            targets (Tensor): ground truth.
            sigma_1 (Tensor): single parameter scaling MSE loss.
            sigma_2 (Tensor): single parameter scaling BCE loss.
        
        Returns:
            Tensor: loss (1,)
        """
        regression_loss = mse_loss(predicted_values, targets)
        classification_loss = binary_cross_entropy_with_logits(predicted_values, (targets > 0).float())

        return regression_loss/(2*pow(sigma_1, 2)) + classification_loss/(pow(sigma_2, 2)) + log(sigma_1) + log(sigma_2)
    
        