"""EEG Models Package"""

from .eegnet import EEGNet
from .base_eegnet import BaseEEGNet, BaseEEGNetGRL
from .eegnetDRO import EEGNetDRO
from .eegnet_grl import EEGNetGRL, EEGNetLNL, EEGNetMI, EEGNetLNLAutoCorrelation
from .eegnet_grl_lag import EEGNetLNLLag

# DMMR Models (with error handling for missing dependencies)
try:
    from .dmmr_adapter import DMMRAdapter, DMMRPreTraining, DMMRFineTuning
    _DMMR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DMMR models not available: {e}")
    _DMMR_AVAILABLE = False

__all__ = [
    'EEGNet',
    'BaseEEGNet', 
    'BaseEEGNetGRL',
    'EEGNetDRO',
    'EEGNetGRL',
    'EEGNetLNL', 
    'EEGNetMI',
    'EEGNetLNLAutoCorrelation',
    'EEGNetLNLLag'
]

# Add DMMR models to __all__ if available
if _DMMR_AVAILABLE:
    __all__.extend(['DMMRAdapter', 'DMMRPreTraining', 'DMMRFineTuning'])