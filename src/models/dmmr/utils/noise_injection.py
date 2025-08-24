"""
Noise injection methods for DMMR.

Various data augmentation techniques used in DMMR training.
"""
import random
from typing import Any
import torch


def timeStepsShuffle(source_data: torch.Tensor) -> torch.Tensor:
    """
    Time steps shuffling for noise injection.
    Retains the last time step and shuffles the others.
    
    Args:
        source_data: Input tensor [batch_size, time_steps, features]
        
    Returns:
        Shuffled tensor with last time step preserved
    """
    source_data_1 = source_data.clone()
    # Retain the last time step
    curTimeStep_1 = source_data_1[:, -1, :]
    # Get data of other time steps
    dim_size = source_data[:, :-1, :].size(1)
    # Generate a random sequence
    idxs_1 = list(range(dim_size))
    # Generate a shuffled sequence
    random.shuffle(idxs_1)
    # Get data corresponding to the shuffled sequence
    else_1 = source_data_1[:, idxs_1, :]
    # Add the origin last time step
    result_1 = torch.cat([else_1, curTimeStep_1.unsqueeze(1)], dim=1)
    return result_1


def maskTimeSteps(source_data: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Mask certain percentage of time steps (excluding the last one).
    
    Args:
        source_data: Input tensor [batch_size, time_steps, features]
        rate: Masking rate (0.0 to 1.0)
        
    Returns:
        Masked tensor with some time steps set to zero
    """
    source_data_1 = source_data.clone()
    num_zeros = int(source_data.size(1) * rate)
    # Mask certain rate of time steps ignoring the last
    zero_indices_1 = torch.randperm(source_data_1.size(1)-1)[:num_zeros]
    source_data_1[:, zero_indices_1, :] = 0
    return source_data_1


def maskChannels(source_data: torch.Tensor, batch_size: int, time_steps: int, rate: float) -> torch.Tensor:
    """
    Mask certain percentage of channels.
    
    Args:
        source_data: Input tensor [batch_size, time_steps, features]
        batch_size: Batch size
        time_steps: Number of time steps
        rate: Masking rate (0.0 to 1.0)
        
    Returns:
        Masked tensor with some channels set to zero
    """
    # Reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(batch_size, time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    num_zeros = int(source_data_reshaped.size(-1) * rate)
    # Mask certain rate of channels
    zero_indices_1 = torch.randperm(source_data_reshaped_1.size(-1))[:num_zeros]
    source_data_reshaped_1[..., zero_indices_1] = 0
    source_data_reshaped_1 = source_data_reshaped_1.reshape(batch_size, time_steps, 310)
    return source_data_reshaped_1


def shuffleChannels(source_data: torch.Tensor, batch_size: int, time_steps: int) -> torch.Tensor:
    """
    Shuffle channels in the input data.
    
    Args:
        source_data: Input tensor [batch_size, time_steps, features]
        batch_size: Batch size
        time_steps: Number of time steps
        
    Returns:
        Tensor with shuffled channels
    """
    # Reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(batch_size, time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    dim_size = source_data_reshaped[..., :].size(-1)
    # Generate a random sequence
    idxs_1 = list(range(dim_size))
    random.shuffle(idxs_1)
    # Shuffle channels
    source_data_reshaped_1 = source_data_reshaped_1[..., idxs_1]
    result_1 = source_data_reshaped_1.reshape(batch_size, time_steps, 310)
    return result_1