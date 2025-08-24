"""
Data processing utilities for DMMR.

Functions for preparing and processing data for DMMR models.
"""
from typing import Tuple, List
import torch
import numpy as np


def prepare_dmmr_data(
    data: torch.Tensor,
    subjects: torch.Tensor,
    batch_size: int = 10,
    time_steps: int = 6,
    shuffle: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for DMMR training.
    
    Args:
        data: Input data tensor [num_samples, time_steps, features]
        subjects: Subject IDs for each sample
        batch_size: Target batch size
        time_steps: Number of time steps
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (prepared_data, subject_labels)
    """
    if shuffle:
        indices = torch.randperm(data.size(0))
        data = data[indices]
        subjects = subjects[indices]
    
    # Ensure consistent dimensions
    if data.size(1) != time_steps:
        # Pad or truncate time dimension
        if data.size(1) > time_steps:
            data = data[:, :time_steps, :]
        else:
            # Pad with zeros
            padding = torch.zeros(data.size(0), time_steps - data.size(1), data.size(2))
            data = torch.cat([data, padding], dim=1)
    
    return data, subjects


def extract_subject_id_from_filename(filename: str) -> str:
    """
    Extract subject ID from filename.
    
    Args:
        filename: Input filename (e.g., 'PD_1001.npy', 'CTL_1021.csv')
        
    Returns:
        Subject ID string (e.g., 'PD_1001', 'CTL_1021')
    """
    # Remove path and extension
    base_name = filename.split('/')[-1].split('.')[0]
    return base_name


def create_subject_mapping(subject_ids: List[str]) -> dict:
    """
    Create mapping from subject ID strings to numerical indices.
    
    Args:
        subject_ids: List of subject ID strings
        
    Returns:
        Dictionary mapping subject IDs to indices
    """
    unique_subjects = sorted(list(set(subject_ids)))
    return {subject: idx for idx, subject in enumerate(unique_subjects)}