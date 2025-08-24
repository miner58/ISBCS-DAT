# tune_train package

from .base_train_func import train_func, test_func
from .groupDRO_train_func import train_func as groupDRO_train_func

# DMMR functions (with error handling)
try:
    from .DMMR_train_func import DMMR_train_func, test_func as DMMR_test_func
    _DMMR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DMMR train functions not available: {e}")
    _DMMR_AVAILABLE = False

__all__ = ['train_func', 'test_func', 'groupDRO_train_func']

if _DMMR_AVAILABLE:
    __all__.extend(['DMMR_train_func', 'DMMR_test_func'])