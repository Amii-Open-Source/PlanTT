"""
Authors: Nicolas Raymond
         Fatima Davelouis
         Ruchika Verma

Description: Stores the Trainer and EarlyStopper classes used to train the models.           
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from json import dump as json_dump
from numpy import append, array, exp
from numpy import inf as np_inf
from os import remove
from os.path import join
from settings.paths import CHECKPOINTS
from src.data.modules.dataset import EncoderDecoderDataset
from src.data.modules.preprocessing import NUC2ONEHOT_MAPPING
from src.data.modules.transforms import Scaler
from src.utils.loss import Loss
from src.utils.metrics import BinaryClassificationMetric, Metric
from tqdm import tqdm
from torch import autocast, argmax, bernoulli, cat, device, float16, load, LongTensor, matmul, \
no_grad, norm, ones, save, Tensor, where, zeros
from torch.cuda.amp import GradScaler
from torch.distributions import Geometric
from torch.nn import CrossEntropyLoss, L1Loss, Module
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from uuid import uuid4

class Trainer(ABC):
    """
    Abstract Trainer class with common methods and attributes of all other specialized trainer classes.
    """
    def __init__(self, loss_fct_name: str) -> None:
        """
        Sets the name of the loss function and initializes all protected attributes to their default value.
        
        Args:
            loss_fct (str): name of the loss function.
        """
        self._in_training: bool = False
        self._loss_fct_name: str = loss_fct_name
        self._optimizer: Optimizer = None
        self._valid_metric: str = None
        
        # Dictionary that keeps track of metrics evolution
        self._progress: dict[str, dict[str, list[float]]] = {'train': {self._loss_fct_name: []}, 'valid': {self._loss_fct_name: []}}
        
    @abstractmethod
    def _update_weights(self,
                        model: Module,
                        device: device,
                        dataloader: DataLoader,
                        metrics: list[Metric],
                        scaler: Scaler = None) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): neural network.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            scaler (Scaler): object used to scale target values.
            
        Returns:
            Module: optimized model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self,
                 model: Module,
                 device: device,
                 dataloader: DataLoader,
                 metrics: list[Metric],
                 scaler: Scaler = None,
                 **kwargs) -> None:
        """
        Evaluates the current model using the data provided in the given dataloader.

        Args:
            model (Module): neural network.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing test or validation data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            scaler (Scaler): object used to scale target values.
        """
        
        raise NotImplementedError
    
    
    def train(self,
              model: Module,
              device: device,
              datasets: tuple[Dataset, Dataset],
              batch_sizes: tuple[int, int],
              metrics: list[Metric],
              lr: float = 1e-4,
              max_epochs: int = 100,
              patience: int = 50,
              weight_decay: float = 1e-2,
              record_path: str = None,
              scheduler_milestones: list[int] = None,
              scheduler_gamma: int = 0.75,
              return_epoch: bool = False) -> Module | tuple[Module, int]:
        """
        Optimizes the weight of a neural network.
        
        Args:
            model (Module): neural network.
            device (device): device on which to send the model and its inputs.
            datasets (tuple[Dataset, Dataset]): tuple of datasets. The first will be used for training and the second for validation.
            batch_sizes (tuple[int, int]): tuple of batch sizes. The first will be used for training and the second for validation.
            metrics: (list[Metric]): list of metrics. If not empty, the last one will be used for early stopping. Otherwise, the loss function is used.
            lr (float, optional): initial learning rate. Defaults to 1e-4.
            max_epochs (int, optional): maximal number of epochs executed during the training. Defaults to 100.
            patience (int, optional): number of consecutive epochs allowed without validation score improvement. Defaults to 50.
            weight_decay (float, optional): weight associated to L2 penalty in the loss function. Defaults to 1e-2.
            record_path (str, optional): path of the file (w/o extension) containing the scores recorded during the training.
                                         If no path is provided, no file will be saved. Default to None.
            scheduler_milestones (list[int], optional): epochs at which the MultiStepLR scheduler needs to multiply the learning rate by gamma.
                                                        Default to None, which is translated to [75, 100, 125, 150]
            scheduler_gamma (int, optional): value multiply the learning rate at each milestone. Default to 0.5
            return_epoch (bool, optional): if True, the number associated to the last training epoch will be returned. Default to False.

        Returns:
            Module: optimized model.
        """
        # Change the status of the trainer
        self._in_training = True
        
        # Set the optimizer
        self._optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        
        # Set the scheduler
        if scheduler_milestones is None:
            scheduler_milestones = [75]
            while scheduler_milestones[-1] < max_epochs:
                scheduler_milestones.append(scheduler_milestones[-1] + 15)
        
        self._scheduler = MultiStepLR(optimizer=self._optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
        
        # Create the dataloaders
        train_dataloader = DataLoader(datasets[0], batch_sizes[0], shuffle=True)
        valid_dataloader = DataLoader(datasets[1], batch_sizes[1], shuffle=False)
        
        # Save the target scaler
        if hasattr(datasets[0], 'scaler'):
            scaler = datasets[0].scaler
        else:
            scaler = None
        
        # Set the dictionary tracking the progress of the metrics
        for phase in self._progress.keys():
            for metric in metrics:
                self._progress[phase][metric.name] = []
                
        # Set the name of the metric used for the early stopping and initialize the EarlyStopper 
        if len(metrics) != 0:
            self._valid_metric = metrics[-1].name
            early_stopper = EarlyStopper(patience, metrics[-1].to_maximize, record_path)
        else:
            self._valid_metric = self._loss_fct_name
            early_stopper = EarlyStopper(patience, False, record_path)
                
        # Execute the optimization process
        self._epoch = 0
        with tqdm(total=max_epochs, desc='Training') as pbar:
            for i in range(max_epochs):
                
                # Set the epoch attribute
                self._epoch = i
                
                # Update the weights
                self._update_weights(model, device, train_dataloader, metrics, scaler)
                
                # Process to the evaluation on the validation set
                self.evaluate(model, device, valid_dataloader, metrics, scaler)
                
                # Look for early stopping
                early_stopper(self._get_last_valid_score(), model)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(self._get_postfix_dict())
                
                # Stop optimization if patience threshold is reached
                if early_stopper.stop:
                    print(f'\n Training stop at epoch {i} after {patience} epochs without improvement in the validation {self._valid_metric}')
                    break
                
                # Update the scheduler
                self._scheduler.step()
                
        # Extraction of the parameters associated to the best validation score
        model.load_state_dict(early_stopper.get_best_params())
        
        # Removal of the checkpoint file created by the EarlyStopper if no recording path was provided
        if record_path is None:
            early_stopper.remove_checkpoint()
        
        # Saving of the scores recorded during the training
        else:
            with open(f'{record_path}.json', 'w') as file:
                json_dump(self._progress, file, indent=True)
                
        # Change the status of the trainer
        self._in_training = False
        
        if return_epoch:
            return model, self._epoch
        else:
            return model
    
    def _get_last_valid_score(self) -> float:
        """
        Extracts the most recent validation score obtained.
        
        Returns:
            float: most recent validation score
        """
        return self._progress['valid'][self._valid_metric][-1]
    
    def _get_postfix_dict(self) -> dict[str, float]:
        """
        Creates a dictionary containing the latest information required to update the progress bar. 
        
        Returns:
            dict[str, float]: dictionary with current training and validation losses and metric scores.
        """
        # Build dict
        postfix_dict = {f'{metric} ({phase})': f'{self._progress[phase][metric][-1]:.6f}' for phase in ['train', 'valid'] for metric in [self._loss_fct_name, self._valid_metric]}
        postfix_dict['lr'] = f'{self._scheduler.get_last_lr()[0]}'
        
        return postfix_dict
    
class OCMTrainer(Trainer):
    """
    Class encapsulating methods necessary to train an orthologs contrast models (OCM).
    """
    def __init__(self,
                 loss_function: Loss) -> None:
        """
        Sets the loss function.

        Args:
            loss_function (Loss): loss function used for the training.
        """
        self.__loss_function: Loss = loss_function
        super().__init__(loss_fct_name=self.__loss_function.name)
        
    def _update_weights(self,
                        model: Module,
                        device: device,
                        dataloader: DataLoader,
                        metrics: list[Metric],
                        scaler: Scaler = None) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): orthologs contrast model.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            scaler (Scaler): object used to scale target values.
            
        Returns:
            Module: optimized orthologs contrast model.
        """
        # Set the model in training mode and send it to the device
        model.train()
        model.to(device)
        
        # Initialize variables to keep track of the progress of the loss, the predictions, and the targets for one epoch
        total_loss, all_predictions, all_targets  = 0, array([]), array([])
        
        # Set the GradScaler to train with mixed precision
        grad_scaler = GradScaler()
        
        # Execute the batch gradient descent
        for seq_A, seq_B, targets, _ in dataloader:
            
            # Transfer data on the device
            seq_A, seq_B, targets = seq_A.to(device), seq_B.to(device), targets.to(device)
            
            # Clear of the gradient
            self._optimizer.zero_grad()
            
            # Execute the forward pass and compute the loss
            with autocast(device_type=device.type, dtype=float16):
                predictions = model(seq_A, seq_B)
                loss = self.__loss_function(predictions, targets)

            # Update the variables keeping track of the progress
            total_loss += loss.item()
            all_predictions = append(all_predictions, predictions.detach().to('cpu').numpy())
            all_targets = append(all_targets, targets.detach().to('cpu').numpy())
            
            # Do a gradient descent step
            grad_scaler.scale(loss).backward()
            grad_scaler.step(self._optimizer)
            grad_scaler.update()
            
        if not self.__loss_function.is_for_regression:
            
            # During training, since predictions are logits, we need to turn them into probabilities
            all_predictions = 1/(1 + exp(-all_predictions))
            
        # Reverse the scaling made on the targets
        if scaler is not None:
            all_predictions = scaler.apply_inverse_transform(all_predictions)
            all_targets = scaler.apply_inverse_transform(all_targets)
            
        # Update overall performance progression
        self.__update_progress(all_predictions, all_targets, total_loss/len(dataloader), metrics)
            
        return model
    
    def evaluate(self,
                 model: Module,
                 device: device,
                 dataloader: DataLoader,
                 metrics: list[Metric],
                 scaler: Scaler = None) -> None | tuple[Tensor, Tensor, Tensor]:
        """
        Evaluates the current model using the data provided in the given dataloader.

        Args:
            model (Module): orthologs contrast model.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the test or validation data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            scaler (Scaler): object used to scale target values.

        Returns:
            None | tuple[Tensor, Tensor, Tensor]: predictions, targets, and species.
        """
        # Set the model in eval mode and send it to device
        model.eval()
        model.to(device)
        
        # Initialize variables to keep track of the progress of the loss, the predictions and the targets during the evaluation
        total_loss, all_predictions, all_targets, all_species  = 0, array([]), array([]), array([])
        
        with no_grad():
            
            # Proceed to the evaluation
            for seq_A, seq_B, targets, species in dataloader:
                
                # Transfer data on the device
                seq_A, seq_B, targets = seq_A.to(device), seq_B.to(device), targets.to(device)
                
                # Execute the forward pass and compute the loss
                with autocast(device_type=device.type, dtype=float16):
                    predictions = model(seq_A, seq_B)
                    loss = self.__loss_function(predictions, targets)
                
                # Update the variables keeping track of the progress
                total_loss += loss.item()
                all_predictions = append(all_predictions, predictions.detach().to('cpu').numpy())
                all_targets = append(all_targets, targets.detach().to('cpu').numpy())
                all_species = append(all_species, species.to('cpu').numpy())
                
        # Reverse the scaling made on the targets
        if scaler is not None:
            all_predictions = scaler.apply_inverse_transform(all_predictions)
            all_targets = scaler.apply_inverse_transform(all_targets)
                
        # Update overall performance progression (if the trainer is in training mode)
        if self._in_training:
            self.__update_progress(all_predictions, all_targets, total_loss/len(dataloader), metrics, validation=True)
            
        # Other wise, return the scores recorded, the predictions and the targets
        else:
            return all_predictions, all_targets, all_species
            
    def __update_progress(self,
                          predictions: list[float],
                          targets: list[float],
                          mean_epoch_loss: float,
                          metrics: list[Metric],
                          threshold: float = 0.5,
                          validation: bool = False) -> None:
        """
        Computes multiple performance metrics and save them in the '__progress' attribute.
        
        Args:
            predictions (list[float]): all predictions made during an epoch.
            targets (list[float]): targets associated to the predictions calculated during the epoch.
            mean_epoch_loss (float): mean of the loss values obtain for all batches.
            metrics (list[Metric]): list of metrics to measure.
            threshold (float, optional): Threshold used to assign label 1 to an observation (used for classification only). Defaults to 0.5.
            validation (bool, optional): if True, indicates that the metrics were recorded during validation. Defaults to False.
        """
        # Set the correct section name in which to record the scores of the metrics
        section = 'valid' if validation else 'train'
        
        # Record the loss progress
        self._progress[section][self._loss_fct_name].append(mean_epoch_loss)
        
        # Record the score of each metric
        if self.__loss_function.is_for_regression:
            predicted_classes = predictions >= 0
            class_targets = targets >= 0
            for metric in metrics:
                if isinstance(metric, BinaryClassificationMetric):
                    if not metric.from_proba:
                        self._progress[section][metric.name].append(metric(predicted_classes, class_targets))
                else:
                    self._progress[section][metric.name].append(metric(predictions, targets))
                        
        else:
            predicted_classes = predictions >= threshold
            for metric in metrics:
                if metric.from_proba:
                    self._progress[section][metric.name].append(metric(predictions, targets))
                else:
                    self._progress[section][metric.name].append(metric(predicted_classes, targets))
                
class MLMTrainer(Trainer):
    """
    Class encapsulating methods necessary to train a masked language model (MLM).
    """
    SEQ_LENGTH: int = 512  # Number of tokens in a sequence
    
    def __init__(self,
                 pad_token_id: int,
                 mask_token_id: int,
                 masking_percentage: float = 0.15,
                 masking_span: int = 6) -> None:
        """
        Initializes the vocabulary, sets the masking probability, sets the masking span, and then 
        sets other protected attributes to their default values.
        
        Args:
            masking_percentage (float): expected percentage of tokens (that are neither [PAD], [CLS], or [SEP]) 
                                        to be dynamically mask in each sequence during training.
            masking_span (int): expected number of contiguous tokens mask in a sequence.
        """
        super().__init__(loss_fct_name='CE')
        
        if not 0 < masking_percentage < 1:
            raise ValueError('masking_percentage must be in range (0,1)')
        
        if not masking_span > 0:
            raise ValueError('masking_span must be a positive integer.')
        
        self.__pad_token_id: int = pad_token_id
        self.__mask_token_id: int = mask_token_id
        self.__masking_percentage: float = masking_percentage
        self.__masking_span: int = masking_span
        
    def _update_weights(self,
                        model: Module,
                        device: device,
                        dataloader: DataLoader,
                        metrics: list[Metric],
                        scaler: Scaler = None) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): masked language model.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): not used. Only there to match parent's function signature.
            scaler (Scaler): not used. Only there to match parent's function signature.
            
        Returns:
            Module: optimized masked language model.
        """
        # Set the model in training mode
        model.train()
        model.to(device)
        
        # Initialize a variable to keep track of the progress of the loss
        total_loss = 0
        
        # Execute the batch gradient descent
        for token_ids, _ in dataloader:
            
            # Mask token ids
            masked_token_ids, input_mask, padding_filter = self.mask_token_ids(token_ids,
                                                                               self.__masking_percentage,
                                                                               self.__masking_span,
                                                                               self.__pad_token_id,
                                                                               self.__mask_token_id)
      
            # Generate the labels (set to -100 all the tokens that are not masked)
            labels = where(input_mask == 0, -100, token_ids).long()
            
            # Generate attention mask (0 for padding token and 1 otherwise)
            attention_mask = where(padding_filter, 0, ones(masked_token_ids.shape)).long()
            
            # Transfer data on the device
            masked_token_ids, labels, attention_mask = masked_token_ids.to(device), labels.to(device), attention_mask.to(device)
            
            # Clear of the gradient
            self._optimizer.zero_grad()
            
            # Execute the forward pass and compute the loss
            loss = model(input_ids=masked_token_ids, attention_mask=attention_mask, labels=labels).loss
            
            # Update the variables keeping track of the progress
            total_loss += loss.item()
            
            # Do a gradient descent step
            loss.backward()
            self._optimizer.step()
            
        # Update overall performance progression
        self._progress['train'][self._loss_fct_name].append(total_loss/len(dataloader))
            
        return model
    
    def evaluate(self,
                 model: Module,
                 device: device,
                 dataloader: DataLoader,
                 metrics: list[Metric],
                 scaler: Scaler = None) -> None | float:
        """
        Evaluates the current model using the data provided in the given dataloader.

        Args:
            model (Module): masked language model.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing test or validation data.
            metrics (list[Metric]): not used. Only there to match parent's function signature.
            scaler (Scaler): not used. Only there to match parent's function signature.
            
        Returns:
            float: when the method is called outside of the train method, the average loss is returned.
        """
        # Set the model in eval mode
        model.eval()
        model.to(device)
        
        # Initialize a variable to keep track of the progress of the loss
        total_loss = 0
        
        with no_grad():
            
            # Proceed to the evaluation
            for token_ids, input_mask in dataloader:
                
                # Mask token ids and generate labels
                filter = (input_mask == 0)
                masked_token_ids = where(filter, token_ids, self.__mask_token_id)   # Set to [MASK] id all tokens that need to be masked.
                labels = where(filter, -100, token_ids).long()                      # Set to -100 all the tokens that are not masked
                
                # Generate attention mask (0 for padding token and 1 otherwise)
                attention_mask = where(masked_token_ids == self.__pad_token_id, 0, ones(masked_token_ids.shape)).long()
                
                # Transfer data on the device
                masked_token_ids, labels, attention_mask = masked_token_ids.to(device), labels.to(device), attention_mask.to(device)
                
                # Execute the forward pass and compute the loss
                loss = model(input_ids=masked_token_ids, attention_mask=attention_mask, labels=labels).loss
            
                # Update the variables keeping track of the progress
                total_loss += loss.item()
                
        # Update overall performance progression
        if self._in_training:
            self._progress['valid'][self._loss_fct_name].append(total_loss/len(dataloader))
        else:
            return total_loss/len(dataloader)
    
    @staticmethod
    def mask_token_ids(token_ids: LongTensor,
                       masking_percentage: float,
                       masking_span: int,
                       pad_token_id: int,
                       mask_token_id: int) -> tuple[LongTensor, LongTensor, LongTensor]:
        """
        Masks a percentage of token ids that are neither associated to [PAD], [CLS], or [SEP] tokens.
        
        Masking is done by randomly selecting regions of k contiguous token ids to mimic 
        pretraining process of DNABERT.
        
        Args:
            token_ids (LongTensor): batch of sequences of token ids (BATCH SIZE, SEQ LENGTH).
            masking_percentage (float): expected percentage of tokens (that are neither [PAD], [CLS], or [SEP]) masked in each sequence.
            masking_span (int): expected number of contiguous tokens mask in each masked region.
            pad_token_id (int): id of [PAD] token.
            mask_token_id (int): id of [MASK] token.
            
        Returns:
            LongTensor: masked token ids (BATCH SIZE, SEQ LENGTH).
            LongTensor: binary valued tensor indicating tokens that are masked (BATCH SIZE, SEQ LENGTH).
            LongTensor: binary valued tensor indicating padding tokens (BATCH SIZE, SEQ LENGTH).
        """
        
        # Set 'span_start_proba' as the probability of each token to be selected as a starting point in a mask span
        span_start_proba = masking_percentage/masking_span
        mask = ones(token_ids.shape)*span_start_proba
        
        # Zero the probability associated to the [CLS] and [SEP] tokens at the start and end of each sequence respectively
        mask[:, [0, MLMTrainer.SEQ_LENGTH - 1]] = 0
        
        # Zero the probability associated to the padding tokens ([PAD])
        padding_filter = (mask == pad_token_id)
        mask = where(padding_filter, 0, mask)
        
        # Sample span starting point from a bernoulli distribution
        mask = bernoulli(mask).long()
        
        # Mask the next 'masking_span' contiguous tokens
        shift = mask
        for _ in range(masking_span):
            shift = cat((zeros((shift.shape[0], 1)).long(), shift[:, 0:(shift.shape[1]-1)]), 1)
            mask += shift
            
        # Modify mask to make sure all values are binary
        # If two mask spans overlapped there might be values > 1
        mask = where(mask > 1, 1, mask)
        
        # Mask the token_ids
        masked_token_ids = where(mask == 1, mask_token_id, token_ids)
        
        return masked_token_ids, mask, padding_filter
    
    
class EncoderDecoderTrainer(Trainer):
    """
    Class encapsulating methods necessary to pre-train an orthologs contrast model using one-hot encoded sequence.
    """
    def __init__(self,
                 masking_percentage: float = 0.15,
                 average_masking_span: int = 6,
                 max_span: int = 10) -> None:
        """
        Sets the name of the loss function using the parent constructor.
        
        Args:
            masking_percentage (float): expected percentage of nucleotides to be dynamically masked in each sequence during training.
            average_masking_span (int): average masking span sampled from geometric distribution during training epoch.
            max_span (int): maximum masking span used during training epoch.
        """
        super().__init__(loss_fct_name='CE')
        self.__loss_function = CrossEntropyLoss(ignore_index=-100)
        self.__masking_percentage = masking_percentage
        self.__geometric_distribution = Geometric(Tensor([1/(average_masking_span-1)])) 
        self.__max_span = max_span
    
    def _update_weights(self,
                        model: Module,
                        device: device,
                        dataloader: DataLoader,
                        metrics: list[Metric],
                        scaler: Scaler = None) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): encoder-decoder model.
            device (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): not used. Only there to match parent's function signature.
            scaler (Scaler): not used. Only there to match parent's function signature.
            
        Returns:
            Module: optimized orthologs contrast model.
        """
        # Set the model in training mode and send it to device
        model.train()
        model.to(device)
        
        # Initialize variables to keep track of the progress of the loss
        total_loss = 0
        
        # Set the GradScaler to train with mixed precision
        grad_scaler = GradScaler()
            
        # Sample masking span from a geometric distribution shifted by 1 and make sure it stands in the range [1, self.__max_span]
        masking_span = int(min(self.__geometric_distribution.sample().item() + 1, self.__max_span))
        
        # Calculate the probability of a nucleotide to be selected as a starting point for span masking
        span_start_proba = self.__masking_percentage/masking_span
        
        # Execute the batch gradient descent
        for seq in dataloader:
            
            # We generate a mask with the span starting point
            mask = EncoderDecoderDataset.create_mask(seq.shape[0], seq.shape[2], span_start_proba, masking_span)
            
            # We get coordinates that need to be masked and not masked
            masked_coord = mask.nonzero()
            # unmasked_coord = (mask == 0).nonzero()
            
            # We clear memory dedicated to the mask
            del mask
            
            # Transfer data on the device
            seq = seq.to(device)
            
            # Clearing of the gradient
            self._optimizer.zero_grad()
            
            # Execute the forward pass and compute the loss only on nucleotides that were masked
            masked_seq = seq.detach().clone()
            masked_seq[masked_coord[:, 0], :, masked_coord[:, 1], :] = Tensor(NUC2ONEHOT_MAPPING['X']).to(device)
            targets =  argmax(seq.squeeze(dim=1), dim=2)
            # targets[unmasked_coord[:, 0], unmasked_coord[:, 1]] = -100 # Set ignored target to ignore_index (-100). Shape of (N, SEQ LENGTH)
            with autocast(device_type=device.type, dtype=float16):
                loss = self.__loss_function(model(masked_seq), targets)
            
            # Update the variables keeping track of the progress
            total_loss += loss.item()
            
            # Do a gradient descent step
            grad_scaler.scale(loss).backward()
            grad_scaler.step(self._optimizer)
            grad_scaler.update()
            
        # Update overall performance progression
        self._progress['train'][self._loss_fct_name].append(total_loss/len(dataloader))
            
        return model
    
    def evaluate(self,
                 model: Module,
                 device: device,
                 dataloader: DataLoader,
                 metrics: list[Metric],
                 scaler: Scaler = None) -> None | float:
            """
            Evaluates the current model using the data provided in the given dataloader.

            Args:
                model (Module): encoder-decoder model.
                device (device): device on which to send the model and its inputs.
                dataloader (DataLoader): dataloader containing test or validation data.
                metrics (list[Metric]): not used. Only there to match parent's function signature.
                scaler (Scaler): not used. Only there to match parent's function signature.
                
            Returns:
                float: When the method is called outside of the train method, the average loss is returned.
            """
            # Set the model in eval mode
            model.eval()
            model.to(device)
            
            # Initialize a variable to keep track of the progress of the loss
            total_loss = 0
            
            with no_grad():
                
                # Proceed to the evaluation
                for seq, mask in dataloader:
                    
                    # We get coordinates that need to be masked and not masked
                    masked_coord = mask.nonzero()
                    # unmasked_coord = (mask == 0).nonzero()
                    
                    # Transfer data on the device
                    seq = seq.to(device)
                    
                    # Execute the forward pass and compute the loss only on nucleotides that were masked
                    masked_seq = seq.detach().clone()
                    masked_seq[masked_coord[:, 0], :, masked_coord[:, 1], :] = Tensor(NUC2ONEHOT_MAPPING['X']).to(device)
                    targets =  argmax(seq.squeeze(dim=1), dim=2)
                    # targets[unmasked_coord[:, 0], unmasked_coord[:, 1]] = -100 # Set ignored target to ignore_index (-100)
                    with autocast(device_type=device.type, dtype=float16):
                        loss = self.__loss_function(model(masked_seq), targets)
                    
                    # Update the variables keeping track of the progress
                    total_loss += loss.item()
                    
            # Update overall performance progression
            if self._in_training:
                self._progress['valid'][self._loss_fct_name].append(total_loss/len(dataloader))
            else:
                return total_loss/len(dataloader)
            
            
class SelfSupervisedTrainer(Trainer):
    """
    Self supervised trainer optimizing similarity retrieval.
    """
    def __init__(self) -> None:
        """
        Sets the name of the loss function using the parent constructor.
        
        We use the L1 loss because the targets and predictions are all between 0 and 1.
        """
        super().__init__(loss_fct_name='L1loss')
        self.__loss_function = L1Loss()

    def _update_weights(self,
                        model: Module,
                        device: device,
                        dataloader: DataLoader,
                        metrics: list[Metric],
                        scaler: Scaler = None) -> None | float:
            """
            Executes one training epoch.

            Args:
                model (Module): PlanTT tower generating embeddings with positive values
                device (device): device on which to send the model and its inputs.
                dataloader (DataLoader): dataloader containing training data.
                metrics (list[Metric]): not used. Only there to match parent's function signature.
                scaler (Scaler): not used. Only there to match parent's function signature.
                
            Returns:
                float: When the method is called outside of the train method, the average loss is returned.
            """
            # Set the model in training mode and send it to the device
            model.train()
            model.to(device)
        
            # Initialize a variable to keep track of the progress of the loss
            total_loss = 0
        
            # Set the GradScaler to train with mixed precision
            grad_scaler = GradScaler()
        
            # Execute the batch gradient descent
            for seq in dataloader:
                
                # Transfer data on the device
                seq = seq.to(device)
                
                # Clear of the gradient
                self._optimizer.zero_grad()
                
                # Execute the forward pass and compute the loss
                with autocast(device_type=device.type, dtype=float16):
                    predictions = model(seq) # (BATCH SIZE, 1, 3000, 5) -> (BATCH SIZE, EMB SIZE)
                    seq = seq.squeeze(dim=1).reshape(seq.shape[0], -1) # (BATCH SIZE, 1, 3000, 5) -> (BATCH SIZE, 15 000)
                    loss = self.__loss_function(self.__cosine_similarity(predictions), self.__cosine_similarity(seq))
                
                # Update the variables keeping track of the progress
                total_loss += loss.item()
                
                # Do a gradient descent step
                grad_scaler.scale(loss).backward()
                grad_scaler.step(self._optimizer)
                grad_scaler.update()
            
            # Update overall performance progression
            self._progress['train'][self._loss_fct_name].append(total_loss/len(dataloader))
            
            return model
        
    def evaluate(self,
                 model: Module,
                 device: device,
                 dataloader: DataLoader,
                 metrics: list[Metric],
                 scaler: Scaler = None) -> None | float:
            """
            Evaluates the current model using the data provided in the given dataloader.

            Args:
                model (Module): PlanTT tower generating embeddings with positive values
                device (device): device on which to send the model and its inputs.
                dataloader (DataLoader): dataloader containing test or validation data.
                metrics (list[Metric]): not used. Only there to match parent's function signature.
                scaler (Scaler): not used. Only there to match parent's function signature.
                
            Returns:
                float: When the method is called outside of the train method, the average loss is returned.
            """
            # Set the model in training mode and send it to the device
            model.eval()
            model.to(device)
        
            # Initialize a variable to keep track of the progress of the loss
            total_loss = 0
            
            with no_grad():
                
                # Proceed to the evaluation
                for seq in dataloader:
                    
                    # Transfer data on the device
                    seq = seq.to(device)
                    
                    # Execute the forward pass and compute the loss
                    with autocast(device_type=device.type, dtype=float16):
                        predictions = model(seq) # (BATCH SIZE, 1, 3000, 5) -> (BATCH SIZE, EMB SIZE)
                        seq = seq.squeeze(dim=1).reshape(seq.shape[0], -1) # (BATCH SIZE, 1, 3000, 5) -> (BATCH SIZE, 15 000)
                        loss = self.__loss_function(self.__cosine_similarity(predictions), self.__cosine_similarity(seq))
                    
                    # Update the variables keeping track of the progress
                    total_loss += loss.item()
                    
            # Update overall performance progression
            if self._in_training:
                self._progress['valid'][self._loss_fct_name].append(total_loss/len(dataloader))
            else:
                return total_loss/len(dataloader)
            
    def __cosine_similarity(self, x: Tensor) -> Tensor:
        """
        Computes the similarity between sequences of a batch.
        Sequences are considered to only contain POSITIVE VALUES.
        
        The similarity corresponds to the normalized dot product.
        
        Args:
            x (Tensor): batch of sequences (BATCH SIZE, *)
            
        Returns:
            Tensor: similarity scores between sequences (BATCH SIZE*BATCH SIZE, )
        """
        # Calculate norm of sequences
        norms = norm(input=x, p=2, dim=1, keepdim=True) # (BATCH SIZE, *) -> (BATCH SIZE, 1)
        
        # Compute cosine similarities 
        return (matmul(x, x.t())/matmul(norms, norms.t())).flatten()
    

class EarlyStopper:
    """
    Object in charge of monitoring validation score progress.
    """
    def __init__(self,
                 patience: int,
                 maximize: bool,
                 file_path: str = None) -> None:
        """
        Sets private and public attributes and then define the comparison
        method according to the given direction.
        
        Args:
            patience (int): number of epochs without improvement allowed.
            maximize (bool): if True, the metric used for early stopping must be maximized.
            file_path (str, optional): path of the file in which to save the best weights. Default to None.
        """
        # Set private attributes
        self.__patience: int = patience
        self.__counter: int = 0
        self.__path_provided: bool = file_path is not None
        self.__file_path: str = join(CHECKPOINTS, f"{uuid4()}.pt") if not self.__path_provided else f'{file_path}.pt'
        self.__stop: bool = False

        # Set comparison method
        if maximize:
            self.__best_score: float = -np_inf
            self.__is_better: Callable[[float, float], bool] = lambda x, y: x > y
            
        else:
            self.__best_score: float = np_inf
            self.__is_better: Callable[[float, float], bool] = lambda x, y: x < y
            
    @property
    def path_provided(self) -> bool:
        return self.__path_provided
            
    @property
    def stop(self) -> bool:
        return self.__stop

    def __call__(self,
                 score: float,
                 model: Module) -> None:
        """
        Compares the current best validation score against the given one
        and updates the object's attributes.
        
        Args:
            score (float): new validation score.
            model (Module): current model for which we've seen the score.
        """
        # Increment the counter if the score is worst than the best score
        if not self.__is_better(score, self.__best_score):
            self.__counter += 1

            # Change early stopping status if the counter reach the patience
            if self.__counter >= self.__patience:
                self.__stop = True

        # Save the parameters of the model if the score is better than the best score observed before
        else:
            self.__best_score = score
            save(model.state_dict(), self.__file_path)
            self.__counter = 0

    def remove_checkpoint(self) -> None:
        """
        Removes the checkpoint file.
        """
        remove(self.__file_path)

    def get_best_params(self) -> OrderedDict[str, Tensor]:
        """
        Return the last parameters that were save. 
        
        Returns:
            OrderedDict[str, tensor]: state dict containing parameters of the model.
        """
        return load(self.__file_path)