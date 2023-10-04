"""
Authors: Nicolas Raymond

Description: Stores the iterative algorithm aiming to find the single base edits
             that maximizes the expression level of a gene.
"""

from src.data.modules.preprocessing import NUC2ONEHOT_MAPPING, ONEHOT2NUC_MAPPING
from src.models.plantt import PlanTT
from src.utils.analysis import save_editing_heatmap
from torch import argmax, autocast, cat, device, float16, no_grad, stack, Tensor
from torch.nn import Module

VARIATIONS_NB: int = 12000


def get_editing_strategy(plantt: PlanTT,
                         seq: Tensor,
                         dev: device,
                         max_edits: int = 1,
                         batch_size: int = 1,
                         save_figs: bool = True) -> tuple[list[tuple[str, int], float]]:
    """
    Generate a list of singe-base edits to increase the expression level of a gene.

    Args:
        plantt (Module): Instance of the PlanTT architecture.
        seq (Tensor): One-hot encoded sequence of shape (1, 3000, 5) or (3000, 5)
        dev (device): torch device.
        max_edits (int, optional): maximum number of singe-base edits available. Defaults to 1.
        batch_size (int, optional): Batch size used for the forward pass. Default to 1.
        save_figs (bool, optional): if True, figures are saved at each iteration to visualize the edit. Default to True.
    

    Returns:
        tuple[list[tuple[str, int], float]]: list of tuples showing the edits to be made and their location, total mRNA abundance difference predicted
    """
    # Initialize list containing the edits to make and a variable to track the total mRNA abundance change
    edits, total = [], 0
    
    # For each iteration until the reach of the maximal budget of edits
    for i in range(max_edits):
        
        # Predict the impact of each possible single-base modification
        pred = predict_editing_impact(plantt=plantt,
                                      seq=seq,
                                      dev=dev,
                                      batch_size=batch_size)
        
        # Find the single-base modification providing the greatest increase
        arg_max = argmax(pred)
        x, y = (arg_max//pred.shape[1]).item(), (arg_max%pred.shape[1]).item()
        
        
        # If the maximal increase found is negative or equal to 0, we stop the loop
        if pred[x, y] <= 0:
            print(f'Algorithm stopped prematurely at iteration {i} because no more \
                    edits were found to improve transcript abundance')
            break
        
        else:
            
            # Update the total
            total += pred[x, y].item()
            
            # Save the edit
            new_nucleotide = ONEHOT2NUC_MAPPING[x]
            old_nucleotide = ONEHOT2NUC_MAPPING[seq[0, y].nonzero().item()]
            edits.append((f'{old_nucleotide}->{new_nucleotide}', y))
            
            # Modify the sequence
            seq[0, y] = Tensor(NUC2ONEHOT_MAPPING[new_nucleotide])
            
            if save_figs:
                save_editing_heatmap(predictions=pred.numpy(), arg_max=y, iteration=i)
            
    return edits, total
            
            
def generate_sequence_variations(seq: Tensor) -> Tensor:
    """
    Generate all variations of a sequence that can be obtained by replacing a single base.

    Args:
        seq (Tensor): one-hot encoded sequence (1, 3000, 5)

    Returns:
        Tensor: all variations of the sequence (12000, 3000, 5)
    """
    variations = cat([seq.clone().detach() for _ in range(VARIATIONS_NB)])
    for i in range(4):
        for j in range(3000):
            variations[(i*3000)+j, j, :] = 0
            variations[(i*3000)+j, j, i] = 1
            
    return variations.unsqueeze(dim=1)


def predict_editing_impact(plantt: Module,
                           seq: Tensor,
                           dev: device,
                           batch_size: int = 1) -> Tensor:
    """
    Predict the mRNA abundance difference between all single-base variations of a sequence.
    and the sequence itself.
    
    Please be aware that the batch size can slightly affect the embeddings generated (see https://github.com/huggingface/transformers/issues/2401).

    Args:
        plantt (Module): Instance of the PlanTT architecture.
        seq (Tensor): One-hot encoded sequence of shape (1, 3000, 5) or (3000, 5)
        dev (device): torch device.
        batch_size (int, optional): Batch size used for the forward pass. Default to 1.

    Returns:
        Tensor: Predictions of the mRNA abundance difference predicted for all variations. The output is a tensor of shape (4, 3000)
                where each row is respectively associated to nucleotide ACGT and each column represents a position in the 3kb long sequence.
                Hence element at position (0, 1200) represents the impact of having an A at the position 1200.
    """
    # Check the shape of the input sequence
    if len(seq.shape) == 2:
        seq = seq.unsqueeze(dim=0)  
    if seq.shape[0] != 1 or seq.shape[1] != 3000 or seq.shape[2] != 5:
        raise ValueError('Provided sequence must be of shape (1, 3000, 5) or (3000, 5).')

    # Generate all sequence variations
    variations = generate_sequence_variations(seq)
    
    # Set model in eval model and send to device
    plantt.eval()
    plantt.to(dev)
    
    with no_grad():
        
        # Generate the embedding of the original sequence
        if dev.type == 'cuda':
            with autocast(device_type=dev.type, dtype=float16):
                seq_emb = plantt.tower(seq.to(dev)).to('cpu')
        else:
            seq_emb = plantt.tower(seq)
            
            
        # Generate the embedding of the variations and concatenate them into a single tensor
        start_idx = 0
        var_embeddings = []
        if dev.type == 'cuda':
            while start_idx < VARIATIONS_NB:
                batch = variations[start_idx:(start_idx+batch_size)].to(dev)
                with autocast(device_type=dev.type, dtype=float16):
                    var_embeddings.append(plantt.tower(batch).to('cpu'))
                start_idx += batch_size
        else:
            while start_idx < VARIATIONS_NB:
                batch = variations[start_idx:(start_idx+batch_size)]
                var_embeddings.append(plantt.tower(batch))
                start_idx += batch_size
        
        var_embeddings = stack(var_embeddings)  # (12000, 375)
        
        # Subtract the original embedding to all variation embeddings and pass them to the head
        start_idx = 0
        predictions = []
        seq_emb = seq_emb.to(dev)
        if dev.type == 'cuda':
            while start_idx < VARIATIONS_NB:
                batch = var_embeddings[start_idx:(start_idx+batch_size)].to(dev)
                with autocast(device_type=dev.type, dtype=float16):
                    predictions.append(plantt.head(batch, seq_emb).to('cpu'))
                start_idx += batch_size
        else:
            while start_idx < VARIATIONS_NB:
                batch = var_embeddings[start_idx:(start_idx+batch_size)]
                predictions.append(plantt.head(batch, seq_emb))
                start_idx += batch_size
        
        # Reshape the predictions in a grid of shape (4, 3000) where each row is respectively 
        # associated to nucleotides ACGT and each column represents a position in the 3kb long sequence.
        predictions = cat(predictions).reshape(4, 3000)
        
        # Manually set coordinates where the predictions must have been equal to zero
        # Some predictions are slightly different than 0 due to batch size effect on GPU 
        # (see https://github.com/huggingface/transformers/issues/2401)
        zero_coord = seq[0, :, :-1].nonzero(as_tuple=True)
        predictions[zero_coord[1], zero_coord[0]] = 0 
        
        return predictions