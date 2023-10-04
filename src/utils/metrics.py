"""
Authors: Nicolas Raymond

Description: Stores classes associated to performance metrics.         
"""
import sklearn.metrics as skm
from abc import ABC, abstractmethod
from numpy import abs, array, mean
from scipy.stats import spearmanr

class Metric(ABC):
    """
    Abstract class representing a metric.
    """
    def __init__(self,
                 name: str,
                 to_maximize: bool) -> None:
        """
        Sets the private attributes.
        
        Args:
            name (str): name of the metric.
            to_maximize (bool): if True, indicates that the metric must be maximized. Otherwise, it must be minimized.
        """
        super().__init__()
        self.__name: str = name
        self.__to_maximize: bool = to_maximize
        
    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def to_maximize(self) -> bool:
        return self.__to_maximize
    
    @staticmethod
    @abstractmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Computes the metric score.
        
        Args:
            predicted_values (array): values predicted by a model.
            targets (array): ground truth.

        Returns:
            float: metric score
        """
        raise NotImplementedError
    

class BinaryClassificationMetric(Metric):
    """
    Abstract class representing a binary classification metric.
    """
    def __init__(self,
                 name: str,
                 to_maximize: bool,
                 from_proba: bool) -> None:
        """
        Sets the the private attribute 'from_proba'.
        
        Args:
            name (str): name of the metric.
            to_maximize (bool): if True, indicates that the metric must be maximized. Otherwise, it must be minimized.
            from_proba (bool): if True, the metric is calculated using probabilities predicted (not the binary classes).
        """
        super().__init__(name, to_maximize)
        
        # Set the private attribute
        self.__from_proba: bool = from_proba
        
    @property
    def from_proba(self) -> bool:
        return self.__from_proba


class Accuracy(BinaryClassificationMetric):
    """
    Binary accuracy metric.
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='Accuracy', to_maximize=True, from_proba=False)
    
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the ratio of correctly predicted observations.
        (TP + TN)/(TP + FP + TN + FN)
        
        Args:
            predicted_values (array): predicted binary classes.
            targets (array): ground truth (binary classes).

        Returns:
            float: accuracy score.
        """
        return mean(predicted_values == targets)
    
    
class AreaUnderROC(BinaryClassificationMetric):
    """
    Wrapper for the Area Under ROC curve metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='auROC', to_maximize=True, from_proba=True)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the area under ROC curve.
        
        Args:
            predicted_values (array): predicted probabilities of belonging to class 1.
            targets (array): ground truth (binary classes).

        Returns:
            float: area under ROC curve.
        """
        return skm.roc_auc_score(targets, predicted_values)


class BalancedAccuracy(BinaryClassificationMetric):
    """
    Wrapper for the balanced accuracy metric provided by scikit learn.
    
    See : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='BAccuracy', to_maximize=True, from_proba=False)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the balance accuracy, which is:
        (TP/(TP + FN) + TN/(TN + FP))/2 
        
        Args:
            predicted_values (array): predicted binary classes.
            targets (array): ground truth (binary classes).

        Returns:
            float: balanced accuracy
        """
        return skm.balanced_accuracy_score(targets, predicted_values)
            
        
class F1Score(BinaryClassificationMetric):
    """
    Wrapper for the F1-score metric provided by scikit learn.
    
    See : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='F1-score', to_maximize=True, from_proba=False)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the F1-score, which is:
        2 * (precision * recall) / (precision + recall)
        
        Args:
            predicted_values (array): predicted binary classes.
            targets (array): ground truth (binary classes).

        Returns:
            float: F1-score
        """
        return skm.f1_score(targets, predicted_values)
        
    
class Precision(BinaryClassificationMetric):
    """
    Wrapper for the precision metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='Precision', to_maximize=True, from_proba=False)
    
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the precision, which is:
        TP / (TP + FP)
        
        Args:
            predicted_values (array): predicted binary classes.
            targets (array): ground truth (binary classes).

        Returns:
            float: precision.
        """
        return skm.precision_score(targets, predicted_values)
    
    
class Recall(BinaryClassificationMetric):
    """
    Wrapper for the recall metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='Recall', to_maximize=True, from_proba=False)
    
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the precision, which is:
        TP / (TP + FN)
        
        Args:
            predicted_values (array): predicted binary classes.
            targets (array): ground truth (binary classes).

        Returns:
            float: precision.
        """
        return skm.recall_score(targets, predicted_values)


class MeanSquaredError(Metric):
    """
    Wrapper for the mean squared error metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='MSE', to_maximize=False)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the mean squared error.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: mean squared error.
        """
        return skm.mean_squared_error(targets, predicted_values)
    

class RootMeanSquaredError(Metric):
    """
    Wrapper for the root mean squared error metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='RMSE', to_maximize=False)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the root mean squared error.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: root mean squared error.
        """
        return skm.mean_squared_error(targets, predicted_values, squared=False)
    
    
class MeanAbsoluteError(Metric):
    """
    Mean absolute error metric.
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='MAE', to_maximize=False)
    
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the mean absolute error.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: mean absolute error.
        """
        return mean(abs(targets - predicted_values))
    
class MeanAbsolutePercentageError(Metric):
    """
    Wrapper for the mean absolute percentage error metric provided by scikit learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='MAPE', to_maximize=False)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the mean absolute error.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: mean absolute error.
        """
        return skm.mean_absolute_percentage_error(targets, predicted_values)
    
class SpearmanRankCorrelation(Metric):
    """
    Wrapper for the spearman rank correlation metric provided by scipy.
    
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='SpearmanR', to_maximize=True)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the spearman rank correlation.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: spearman rank correlation.
        """
        return spearmanr(targets, predicted_values).correlation
    
class RSquared(Metric):
    """
    Wrapper for the coefficient of determination metric provided by scikit-learn.
    
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='RSquared', to_maximize=True)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the spearman rank correlation.
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: spearman rank correlation.
        """
        return skm.r2_score(targets, predicted_values)
    
    
class AccuracyRMSERatio(Metric):
    """
    Divides the Balanced Accuracy by the Root Mean Squared Error.
    """
    def __init__(self) -> None:
        """
        Sets internal attributes using the parent constructor.
        """
        super().__init__(name='BAcc-RMSE', to_maximize=True)
        
    @staticmethod
    def __call__(predicted_values: array,
                 targets: array) -> float:
        """
        Calculates the balanced accuracy-rmse ratio
        
        Args:
            predicted_values (array): real-valued regression predictions.
            targets (array): ground truth (regression targets).

        Returns:
            float: spearman rank correlation.
        """
        b_acc = skm.balanced_accuracy_score((targets > 0), (predicted_values > 0))
        rmse = skm.mean_squared_error(targets, predicted_values, squared=False)
        
        return (b_acc/rmse)*100
    