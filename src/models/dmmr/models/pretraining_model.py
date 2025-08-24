"""
DMMR Pre-training Model.

Domain-invariant representation learning with mixed reconstruction.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Attention, Encoder, Decoder, DomainClassifier, ReverseLayerF
from ..utils import MSE, timeStepsShuffle


class DMMRPreTrainingModel(nn.Module):
    """
    DMMR Pre-training Model.
    
    DMMR 사전훈련 모델로 도메인 불변 표현 학습과 
    혼합 재구성을 통한 robust representation을 학습합니다.
    
    Args:
        number_of_source: 소스 도메인 수 (default: 14, 피험자 수)
        number_of_category: 클래스 수 (default: 3)  
        batch_size: 배치 크기 (default: 10)
        time_steps: 시간 단계 수 (default: 15)
        input_dim: 입력 차원 (default: 310, EEG features)
    """
    
    def __init__(
        self,
        number_of_source: int = 14,
        number_of_category: int = 3,
        batch_size: int = 10,
        time_steps: int = 15,
        input_dim: int = 310
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.input_dim = input_dim
        # Core components
        self.attention_layer = Attention(input_dim=input_dim)
        self.shared_encoder = Encoder(input_dim=input_dim, hid_dim=64, n_layers=1)
        self.domain_classifier = DomainClassifier(input_dim=64, output_dim=number_of_source)
        
        # Loss functions
        self.mse = MSE()
        
        # Decoders for each source domain (안전한 구현)
        self.decoders = nn.ModuleList([
            Decoder(input_dim=input_dim, hid_dim=64, n_layers=1, output_dim=input_dim)
            for _ in range(number_of_source)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        corres: torch.Tensor, 
        subject_id: torch.Tensor, 
        m: float = 0.0, 
        mark: int = 0,
        subject_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DMMR Pre-training forward pass.
        
        Args:
            x: Input data [batch_size, time_steps, input_dim]
            corres: Correspondence data for reconstruction supervision
            subject_id: Subject IDs for domain classification
            m: Gradient reversal multiplier (default: 0.0)
            mark: Training phase marker (default: 0)
            subject_mask: Subject mask data for single-label subject data case
            
        Returns:
            Tuple of (reconstruction_loss, similarity_loss)
        """
        # Noise Injection: Time Steps Shuffling
        x = timeStepsShuffle(x)
        
        # Attention-Based Pooling (ABP) module
        x = self.attention_layer(x, x.shape[0], self.time_steps)
        
        # Encode weighted features with shared encoder
        shared_last_out, shared_hn, shared_cn = self.shared_encoder(x)
        
        # Domain Adversarial Training (DG_DANN module)
        # Gradient Reversal Layer
        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        
        # Subject/Domain Discriminator
        subject_predict = self.domain_classifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict, dim=1)
        
        # Domain adversarial loss
        sim_loss = F.nll_loss(subject_predict, subject_id)
        
        # Build supervision for decoders
        # print(f"🔍 Correspondence data shape: {corres.shape}")
        # print(f"🔍 Expected batch_size from corres.shape[0]: {corres.shape[0]}")
        # print(f"🔍 Input x shape: {x.shape}")
        # print(f"🔍 Model time_steps: {self.time_steps}")
        
        # Attention layer로 correspondence 데이터 처리
        # subject_mask 값으로 correspondence 데이터 필터링
        # corres: (batch_size*subject_num, time_steps, features), subject_mask: (batch_size, subject_num)
        splitted_corres = torch.chunk(corres, self.number_of_source, dim=0)

        # subject_mask: (batch_size, subject_num)
        # 배치 전체에서 하나라도 유효하면 해당 subject는 유효
        # batch_size의 subject별 mask 값은 같음
        valid_subjects_mask = torch.any(subject_mask > 0, dim=0)  # (subject_num,)
        valid_subject_indices = torch.where(valid_subjects_mask)[0]

        # 3단계: 유효한 correspondence만 선택
        valid_corres_chunks = [splitted_corres[i] for i in valid_subject_indices]
        
        # 4단계: 유효한 correspondence만 concatenate하여 attention 적용
        valid_corres = torch.cat(valid_corres_chunks, dim=0)
        # print(f"🔍 Valid correspondence shape before attention: {valid_corres.shape}")
        
        # Attention은 유효한 데이터에만 적용
        valid_corres = self.attention_layer(valid_corres, valid_corres.shape[0], self.time_steps)
        # print(f"🔍 Valid correspondence shape after attention: {valid_corres.shape}")

        # 다시 subject별로 분할
        splitted_tensors = torch.chunk(valid_corres, len(valid_subject_indices), dim=0)

        # First stage: Reconstruct features and create mixed features
        rec_loss = 0
        mix_subject_feature = 0

        # subject_mask가 1인 decoder만을 사용
        for i, subject_idx in enumerate(valid_subject_indices):
            # Reconstruct features in first stage
            x_out, *_ = self.decoders[subject_idx](shared_last_out, shared_hn, shared_cn, self.time_steps)
            # Mix method for data augmentation
            mix_subject_feature += x_out
        
        # for i, decoder in enumerate(self.decoders):
        #     # Reconstruct features in first stage
        #     x_out, *_ = decoder(shared_last_out, shared_hn, shared_cn, self.time_steps)
        #     # Mix method for data augmentation
        #     mix_subject_feature += x_out
        
        # Second stage: Re-encode mixed features
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.shared_encoder(mix_subject_feature)
        
        # Second stage: Reconstruct and compute loss
        # subject_mask가 1인 decoder만을 사용
        for i, subject_idx in enumerate(valid_subject_indices):
            # Reconstruct features in second stage
            x_out, *_ = self.decoders[subject_idx](shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            # Compute reconstruction loss only in second stage
            rec_loss += self.mse(x_out, splitted_tensors[i])

        # for i, decoder in enumerate(self.decoders):
            # x_out, *_ = decoder(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            # # Compute reconstruction loss only in second stage
            # rec_loss += self.mse(x_out, splitted_tensors[i])
        
        return rec_loss, sim_loss