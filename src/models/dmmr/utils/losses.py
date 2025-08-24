"""
Loss functions for DMMR.

Custom loss functions used in DMMR training.
"""
import torch
import torch.nn as nn


class MSE(nn.Module):
    """
    Mean Squared Error loss function.
    
    Custom MSE implementation used in DMMR for reconstruction loss.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Args:
            pred: Predicted values
            real: Ground truth values
            
        Returns:
            MSE loss value
        """
        if pred.shape != real.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs real {real.shape}")
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse