"""
DMMR Base Lightning Module.

Provides base Lightning module functionality for DMMR models.
"""
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


class DMMRBaseLightningModule(pl.LightningModule, ABC):
    """
    Abstract base class for all DMMR Lightning modules.
    
    DMMR Lightning 모듈들의 추상 기본 클래스입니다.
    하이퍼파라미터 관리, 메트릭 시스템, 옵티마이저 설정,
    표준화된 로깅 등의 공통 기능을 제공합니다.
    
    Args:
        optimizer_type: 옵티마이저 타입 (default: 'adam')
        optimizer_config: 옵티마이저 설정 (선택적)
        scheduler_type: 스케줄러 타입 (선택적)
        scheduler_config: 스케줄러 설정 (선택적)
        num_classes: 클래스 수 (default: 3)
        input_dim: 입력 차원 (default: 310)
        hidden_dim: 히든 차원 (default: 64)
        batch_size: 배치 크기 (default: 10)
        time_steps: 시간 단계 수 (default: 15)
        beta: 손실 균형 파라미터 (default: 1.0)
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        optimizer_type: str = 'adam',
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_type: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 3,
        input_dim: int = 310,
        hidden_dim: int = 64,
        batch_size: int = 10,
        time_steps: int = 15,
        beta: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters automatically
        self.save_hyperparameters()
        
        # DMMR-specific parameters
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.optimizer_config = optimizer_config or {}
        self.scheduler_type = scheduler_type
        self.scheduler_config = scheduler_config or {}
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.beta = beta
        
        # Setup metrics
        self._setup_metrics()
        
        # Initialize model components (implemented by subclasses)
        self._build_model()
    
    def _setup_metrics(self) -> None:
        """Setup metrics for train/val/test phases."""
        # Classification metrics
        metric_dict = {
            "macro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_classes, average='macro'
            ),
            "macro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=self.num_classes, average='macro'
            ),
            "macro_precision": torchmetrics.classification.Precision(
                task="multiclass", num_classes=self.num_classes, average='macro'
            ),
            "macro_recall": torchmetrics.classification.Recall(
                task="multiclass", num_classes=self.num_classes, average='macro'
            ),
            "micro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=self.num_classes, average='micro'
            ),
            "micro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=self.num_classes, average='micro'
            ),
            "micro_precision": torchmetrics.classification.Precision(
                task="multiclass", num_classes=self.num_classes, average='micro'
            ),
            "micro_recall": torchmetrics.classification.Recall(
                task="multiclass", num_classes=self.num_classes, average='micro'
            ),
        }
        
        # Create metric collections
        self.train_metrics = torchmetrics.MetricCollection(
            metric_dict, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build model architecture - implemented by subclasses."""
        pass
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """Configure optimizer and scheduler."""
        # Configure optimizer
        if self.optimizer_type.lower() == 'adam':
            optimizer = Adam(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_config
            )
        elif self.optimizer_type.lower() == 'sgd':
            optimizer = SGD(
                self.parameters(), 
                lr=self.learning_rate,
                **self.optimizer_config
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        
        # Configure scheduler if specified
        if self.scheduler_type is None:
            return optimizer
        
        if self.scheduler_type.lower() == 'steplr':
            scheduler = StepLR(optimizer, **self.scheduler_config)
        elif self.scheduler_type.lower() == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(optimizer, **self.scheduler_config)
        elif self.scheduler_type.lower() == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='max',
                **self.scheduler_config
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_macro_acc',
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
        
        return [optimizer], [scheduler]
    
    def _compute_and_log_metrics(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        stage: str
    ) -> Dict[str, torch.Tensor]:
        """Compute and log metrics."""
        # Get appropriate metrics
        if stage == 'train':
            metrics = self.train_metrics(preds, targets)
        elif stage == 'val':
            metrics = self.val_metrics(preds, targets)
        elif stage == 'test':
            metrics = self.test_metrics(preds, targets)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=(metric_name.endswith('_macro_acc')),
                sync_dist=True
            )
        
        return metrics
    
    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step - implemented by subclasses."""
        pass
    
    @abstractmethod 
    def validation_step(self, batch, batch_idx):
        """Validation step - implemented by subclasses."""
        pass
    
    def test_step(self, batch, batch_idx):
        """Test step - default uses validation step."""
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
            
        with torch.no_grad():
            outputs = self(x)
            if isinstance(outputs, (list, tuple)):
                predictions = outputs[0]
            else:
                predictions = outputs
                
        return torch.softmax(predictions, dim=-1)