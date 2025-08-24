"""
DMMR Package.

Domain-invariant representation learning for robust EEG classification.
"""
# Core models
from .models import (
    DMMRPreTrainingModel,
    DMMRFineTuningModel,
    DMMRTestModel
)

# Lightning modules
from .modules import (
    DMMRBaseLightningModule,
    DMMRPreTrainingModule,
    DMMRFineTuningModule
)

# Layers
from .layers import (
    Attention,
    LSTM,
    Encoder,
    Decoder,
    DomainClassifier,
    ReverseLayerF,
    gradient_reverse
)

# Utilities
from .utils import (
    timeStepsShuffle,
    maskTimeSteps,
    maskChannels,
    shuffleChannels,
    MSE,
    prepare_dmmr_data
)

# Compatibility aliases for existing code
DMMRPreTraining = DMMRPreTrainingModule
DMMRFineTuning = DMMRFineTuningModule

__all__ = [
    # Core models
    'DMMRPreTrainingModel',
    'DMMRFineTuningModel', 
    'DMMRTestModel',
    
    # Lightning modules
    'DMMRBaseLightningModule',
    'DMMRPreTrainingModule',
    'DMMRFineTuningModule',
    
    # Compatibility aliases
    'DMMRPreTraining',
    'DMMRFineTuning',
    
    # Layers
    'Attention',
    'LSTM',
    'Encoder', 
    'Decoder',
    'DomainClassifier',
    'ReverseLayerF',
    'gradient_reverse',
    
    # Utilities
    'timeStepsShuffle',
    'maskTimeSteps',
    'maskChannels', 
    'shuffleChannels',
    'MSE',
    'prepare_dmmr_data'
]