"""
DMMR Lightning Model Adapter

Provides seamless integration with existing model registration and Ray Tune workflows.
Uses the new unified DMMR structure in src/models/dmmr/.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, Union

# Import unified DMMR components
from .dmmr import (
    DMMRPreTrainingModule,
    DMMRFineTuningModule
)


class DMMRPreTraining(DMMRPreTrainingModule):
    """
    DMMR Pre-training adapter for compatibility with existing model registration.
    
    Provides parameter compatibility with EEGNet-style parameters while
    using DMMR-specific architecture internally.
    """
    
    def __init__(
        self,
        # Standard EEGNet-compatible parameters
        nb_classes: int = 2,
        Chans: int = 32,
        Samples: int = 256,
        kernLength: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropoutRate: float = 0.5,
        lr: float = 1e-3,
        class_weight: Optional[list] = None,
        
        # DMMR-specific parameters
        number_of_source: int = 3,
        batch_size: int = 10,
        time_steps: int = 6,
        input_dim: int = 310,
        hidden_dim: int = 64,
        beta: float = 1.0,
        
        # Training parameters
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = None,
        
        **kwargs
    ):
        # Map EEGNet parameters to DMMR parameters
        dmmr_config = {
            'learning_rate': lr,
            'optimizer_type': optimizer_type,
            'scheduler_type': scheduler_type,
            'num_classes': nb_classes,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'time_steps': time_steps,
            'beta': beta,
            'number_of_source': number_of_source,
            **kwargs
        }
        
        super().__init__(**dmmr_config)
        
        # Store compatibility parameters
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.class_weight = class_weight


class DMMRFineTuning(DMMRFineTuningModule):
    """
    DMMR Fine-tuning adapter for compatibility with existing model registration.
    
    Inherits from pre-training and adds classification capabilities.
    """
    
    def __init__(
        self,
        # Standard EEGNet-compatible parameters
        nb_classes: int = 2,
        Chans: int = 32,
        Samples: int = 256,
        kernLength: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropoutRate: float = 0.0,
        lr: float = 1e-4,  # Lower learning rate for fine-tuning
        class_weight: Optional[list] = None,
        
        # DMMR fine-tuning parameters
        pretrained_module=None,
        pretrained_checkpoint_path: Optional[str] = None,
        freeze_pretrained: bool = True,
        number_of_source: int = 3,
        batch_size: int = 10,
        time_steps: int = 6,
        input_dim: int = 310,
        hidden_dim: int = 64,
        
        # Training parameters
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = None,
        
        **kwargs
    ):
        # Map EEGNet parameters to DMMR parameters
        dmmr_config = {
            'learning_rate': lr,
            'optimizer_type': optimizer_type,
            'scheduler_type': scheduler_type,
            'num_classes': nb_classes,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'time_steps': time_steps,
            'pretrained_module': pretrained_module,
            'pretrained_checkpoint_path': pretrained_checkpoint_path,
            'freeze_pretrained': freeze_pretrained,
            'number_of_source': number_of_source,
            'dropout_rate': dropoutRate,
            **kwargs
        }
        
        super().__init__(**dmmr_config)
        
        # Store compatibility parameters
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.class_weight = class_weight


# Legacy adapter class for backward compatibility
class DMMRAdapter(pl.LightningModule):
    """
    Legacy adapter class for backward compatibility.
    
    Automatically determines whether to use pre-training or fine-tuning
    based on the provided parameters.
    """
    
    def __init__(
        self,
        mode: str = "pretraining",  # "pretraining" or "finetuning"
        **kwargs
    ):
        super().__init__()
        
        self.mode = mode.lower()
        
        if self.mode == "pretraining":
            self.model = DMMRPreTraining(**kwargs)
        elif self.mode == "finetuning":
            self.model = DMMRFineTuning(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pretraining' or 'finetuning'")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()


def create_dmmr_model(
    model_type: str = "pretraining",
    **config
) -> Union[DMMRPreTraining, DMMRFineTuning]:
    """
    Factory function to create DMMR models.
    
    Args:
        model_type: "pretraining" or "finetuning"
        **config: Model configuration parameters
        
    Returns:
        Configured DMMR model
    """
    if model_type.lower() == "pretraining":
        return DMMRPreTraining(**config)
    elif model_type.lower() == "finetuning":
        return DMMRFineTuning(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# For compatibility with existing imports
__all__ = [
    'DMMRPreTraining',
    'DMMRFineTuning', 
    'DMMRAdapter',
    'create_dmmr_model'
]