"""
DMMR Test Model.

Test model for inference using finetuned DMMR representations.
"""
import torch
import torch.nn as nn
import copy


class DMMRTestModel(nn.Module):
    """
    DMMR Test Model.
    
    DMMR 테스트 모델로 파인튜닝된 모델을 사용하여
    추론을 수행합니다.
    
    Args:
        base_model: 파인튜닝된 DMMR 모델
    """
    
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DMMR Test forward pass.
        
        Args:
            x: Input data [batch_size, time_steps, input_dim]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Apply attention-based pooling
        x = self.base_model.attention_layer(
            x, 
            self.base_model.batch_size, 
            self.base_model.time_steps
        )
        
        # Extract features using shared encoder
        shared_last_out, shared_hn, shared_cn = self.base_model.shared_encoder(x)
        
        # Classification
        x_shared_logits = self.base_model.cls_fc(shared_last_out)
        
        return x_shared_logits