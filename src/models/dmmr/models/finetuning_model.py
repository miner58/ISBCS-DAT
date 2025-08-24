"""
DMMR Fine-tuning Model.

Fine-tuning model for emotion/task classification using pretrained representations.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..layers import Attention, Encoder, Decoder
from ..utils import MSE


class DMMRFineTuningModel(nn.Module):
    """
    DMMR Fine-tuning Model.
    
    DMMR 파인튜닝 모델로 사전훈련된 표현을 사용하여
    감정/과제 분류를 수행합니다.
    
    Args:
        base_model: 사전훈련된 DMMR 모델 (선택적)
        number_of_source: 소스 도메인 수 (default: 14)
        number_of_category: 클래스 수 (default: 3)
        batch_size: 배치 크기 (default: 10)
        time_steps: 시간 단계 수 (default: 15)
        input_dim: 입력 차원 (default: 310)
        hidden_dim: 히든 차원 (default: 64)
        n_layers: LSTM 레이어 수 (default: 1)
        dropout_rate: 드롭아웃 비율 (default: 0.5)
        num_classes: 분류 클래스 수 (선택적)
    """
    
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        number_of_source: int = 14,
        number_of_category: int = 3,
        batch_size: int = 10,
        time_steps: int = 15,
        input_dim: int = 310,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout_rate: float = 0.0,
        num_classes: Optional[int] = None
    ) -> None:
        super().__init__()
        
        # Store parameters
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes or number_of_category
        
        if base_model is not None:
            # Use pretrained model components
            self.base_model = copy.deepcopy(base_model)
            self.attention_layer = self.base_model.attention_layer
            self.shared_encoder = self.base_model.shared_encoder
        else:
            # Create new components (for standalone use)
            self.attention_layer = Attention(input_dim=self.input_dim)
            self.shared_encoder = Encoder(
                input_dim=self.input_dim, 
                hid_dim=self.hidden_dim, 
                n_layers=self.n_layers
            )
            
        # Classification head for emotion/task recognition
        self.cls_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), 
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes, bias=True)
        )
        
        # Loss functions
        self.mse = MSE()
        
        # Decoders for reconstruction (안전한 구현)
        self.decoders = nn.ModuleList([
            Decoder(
                input_dim=self.input_dim, 
                hid_dim=self.hidden_dim, 
                n_layers=self.n_layers, 
                output_dim=self.input_dim
            )
            for _ in range(number_of_source)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        label_src: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DMMR Fine-tuning forward pass.
        
        Args:
            x: Input data [batch_size, time_steps, input_dim]
            label_src: Target labels for classification (선택적)
            
        Returns:
            Tuple of (x_pred, x_logits, cls_loss)
        """
        # Apply attention-based pooling
        x = self.attention_layer(x, x.shape[0], self.time_steps)
        
        # Extract features using shared encoder
        shared_last_out, shared_hn, shared_cn = self.shared_encoder(x)
        
        # Classification
        x_logits = self.cls_fc(shared_last_out)
        x_pred = F.log_softmax(x_logits, dim=1)
        
        # Compute classification loss if labels provided
        cls_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if label_src is not None:
            cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        
        return x_pred, x_logits, cls_loss