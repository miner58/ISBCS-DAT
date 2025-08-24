"""
DMMR Pre-training Lightning Module.

Implements pre-training phase with reconstruction loss and domain adversarial training.
"""
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from .lightning_base import DMMRBaseLightningModule
from ..models.pretraining_model import DMMRPreTrainingModel


class DMMRPreTrainingModule(DMMRBaseLightningModule):
    """
    DMMR Pre-training Lightning Module.
    
    DMMR 사전훈련 Lightning 모듈로 다음 기능들을 포함합니다:
    - 다중 디코더를 이용한 재구성 손실
    - 그래디언트 역전을 통한 도메인 적대적 훈련
    - 노이즈 주입 및 혼합 증강
    - 다중 소스 도메인 처리
    
    Args:
        number_of_source: 소스 도메인 수 (default: 14)
        noise_injection_type: 노이즈 주입 타입 ("shuffle", "mask", "none")
        noise_rate: 노이즈 비율 (default: 0.2)
    """
    
    def __init__(
        self,
        number_of_source: int = 14,
        noise_injection_type: str = "shuffle",
        noise_rate: float = 0.2,
        **kwargs
    ):
        # DMMR 기본값 설정
        dmmr_defaults = {
            'num_classes': 3,
            'input_dim': 310,
            'hidden_dim': 64,
            'batch_size': 10,
            'time_steps': 15,
            'beta': 1.0,
        }
        
        # 사전훈련 특화 파라미터를 먼저 설정
        self.number_of_source = number_of_source
        self.noise_injection_type = noise_injection_type
        self.noise_rate = noise_rate
        
        config = {**dmmr_defaults, **kwargs}
        super().__init__(**config)
        
        # 🔧 핵심 변경: DMMR은 항상 Manual optimization
        self.automatic_optimization = False
        self.subject_iterators = {}
        
        # 사전훈련 특화 메트릭 설정
        self._setup_pretraining_metrics()
    
    def _build_model(self) -> None:
        """DMMR 사전훈련 모델 구조 생성."""
        self.model = DMMRPreTrainingModel(
            number_of_source=self.number_of_source,
            number_of_category=self.num_classes,
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            input_dim=self.input_dim
        )
    
    def _setup_pretraining_metrics(self) -> None:
        """사전훈련 특화 메트릭 설정."""
        # 손실 메트릭
        self.train_loss_metrics = torchmetrics.MetricCollection({
            "total_loss": torchmetrics.MeanMetric(),
            "rec_loss": torchmetrics.MeanMetric(),
            "sim_loss": torchmetrics.MeanMetric(),
        }, prefix="train_")
        
        self.val_loss_metrics = self.train_loss_metrics.clone(prefix="val_")
        
        # 도메인 적대적 정확도
        self.domain_accuracy = torchmetrics.Accuracy(
            task='multiclass', 
            num_classes=self.number_of_source
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        corres: torch.Tensor, 
        subject_id: torch.Tensor, 
        m: float = 0.0,
        subject_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pre-training.
        
        Args:
            x: Input EEG data
            corres: Correspondence data for reconstruction
            subject_id: Subject/domain IDs
            m: Gradient reversal strength
            
        Returns:
            Tuple of (reconstruction_loss, similarity_loss)
        """
        return self.model.forward(x, corres, subject_id, m, mark=0, subject_mask=subject_mask)
    
    def training_step(self, batch, batch_idx):
        """DMMR 원본 방식: 4D 데이터에서 subject_num 축 기준 피험자별 순환 훈련 (Manual Optimization)"""
        
        # 🔧 핵심 변경: Manual optimization
        optimizer = self.optimizers()
        
        # DataModule 출력: (source_data, correspondence_data, subject_ids, source_labels, subject_mask_data)
        source_data, correspondence_data, subject_ids, source_labels, subject_mask_data = batch
        batch_size, subject_num, time_steps, features = source_data.shape
        
        total_loss = 0.0
        total_rec_loss = 0.0
        total_sim_loss = 0.0
        num_processed = 0
        
        # GRL 강도 계산 (원본 DMMR과 동일)
        grl_strength = self._calculate_grl_strength()
        
        print(f"🔍 Training batch shapes: source_data={source_data.shape}, correspondence_data={correspondence_data.shape}")
        print(f"🔍 Subject count: {subject_num}, batch_size: {batch_size}")
        
        # 🎯 핵심: subject_num 축 기준으로 피험자별 순환 처리 (원본 DMMR 방식)
        for subject_idx in range(subject_num):
            try:
                # 현재 피험자의 데이터 추출 (축 기준 슬라이싱)
                subject_batch_data = source_data[:, subject_idx, :, :]  # (batch_size, time_steps, features)
                subject_correspondence = correspondence_data[subject_idx,:, :, :]  # (batch_size*subject_nums,time_steps, features)
                subject_labels = source_labels[:, subject_idx]  # (batch_size,)
                subject_mask = subject_mask_data[subject_idx, :, :]  # (batch_size, subject_num)
                
                # 피험자 ID 텐서 생성 (원본 DMMR 방식: 모든 배치에서 동일한 subject_idx)
                current_subject_ids = torch.full(
                    (batch_size,), 
                    subject_idx, 
                    dtype=torch.long, 
                    device=source_data.device
                )
                
                # DMMR forward (원본과 동일한 호출 방식)
                rec_loss, sim_loss = self.forward(
                    subject_batch_data, subject_correspondence, 
                    current_subject_ids, m=grl_strength, subject_mask=subject_mask
                )
                
                # Loss 계산 (원본 DMMR과 동일)
                subject_loss = rec_loss + self.beta * sim_loss
                
                # 🔧 핵심: Manual backward (원본 DMMR 방식 - 피험자별 독립적 backward)
                self.manual_backward(subject_loss)
                
                # Loss 누적
                total_loss += subject_loss.detach()
                total_rec_loss += rec_loss.detach()
                total_sim_loss += sim_loss.detach()
                num_processed += 1
                
                # Debug info (첫 번째 배치의 첫 번째 피험자에서만)
                if batch_idx == 0 and subject_idx == 0:
                    print(f"🔄 Subject {subject_idx}: rec_loss={rec_loss:.4f}, sim_loss={sim_loss:.4f}")
                    print(f"🔍 Subject shapes: data={subject_batch_data.shape}, correspondence={subject_correspondence.shape}, labels={subject_labels.shape}")
                
            except Exception as e:
                print(f"⚠️ Subject {subject_idx} 처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Optimizer step (모든 피험자 처리 후 - 원본 DMMR 방식)
        if num_processed > 0:
            optimizer.step()
            optimizer.zero_grad()
            
            avg_total_loss = total_loss / num_processed
            avg_rec_loss = total_rec_loss / num_processed
            avg_sim_loss = total_sim_loss / num_processed
        else:
            avg_total_loss = torch.tensor(0.0, device=self.device)
            avg_rec_loss = torch.tensor(0.0, device=self.device)
            avg_sim_loss = torch.tensor(0.0, device=self.device)
        
        # 메트릭 업데이트
        self.train_loss_metrics['total_loss'].update(
            avg_total_loss.item() if isinstance(avg_total_loss, torch.Tensor) else avg_total_loss
        )
        self.train_loss_metrics['rec_loss'].update(
            avg_rec_loss.item() if isinstance(avg_rec_loss, torch.Tensor) else avg_rec_loss
        )
        self.train_loss_metrics['sim_loss'].update(
            avg_sim_loss.item() if isinstance(avg_sim_loss, torch.Tensor) else avg_sim_loss
        )
        
        # 로깅
        self.log_dict({
            'train_total_loss': avg_total_loss,
            'train_rec_loss': avg_rec_loss,
            'train_sim_loss': avg_sim_loss,
            'grl_strength': grl_strength,
            'subjects_processed': num_processed,
            'subject_num': subject_num,
            'batch_size': batch_size
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return avg_total_loss
    
    def _get_subject_batch(self, subject_id: str, subject_dataloader):
        """Subject별 배치 가져오기 (원본 DMMR 방식)"""
        
        # Iterator 관리 (원본 DMMR과 동일)
        if subject_id not in self.subject_iterators:
            self.subject_iterators[subject_id] = iter(subject_dataloader)
        
        try:
            # Subject에서 배치 가져오기
            batch = next(self.subject_iterators[subject_id])
            return batch
        except StopIteration:
            # Iterator 재시작 (원본 DMMR과 동일)
            self.subject_iterators[subject_id] = iter(subject_dataloader)
            try:
                batch = next(self.subject_iterators[subject_id])
                return batch
            except StopIteration:
                return None
    
    def _calculate_grl_strength(self) -> float:
        """GRL 강도 계산 (원본 DMMR과 동일)"""
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches > 0:
            progress = self.trainer.global_step / max(1, self.trainer.estimated_stepping_batches)
        else:
            progress = min(1.0, self.current_epoch / 100.0)
        
        # 원본 DMMR 공식: 2.0 / (1.0 + exp(-10 * progress)) - 1.0
        m = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0
        return float(m)
    
    def validation_step(self, batch, batch_idx):
        """DMMR 원본 방식: 4D 데이터에서 subject_num 축 기준 피험자별 순환 Validation"""

        # DataModule 출력: (source_data, correspondence_data, subject_ids, source_labels, subject_mask_data)
        source_data, correspondence_data, _, source_labels, subject_mask_data = batch
        batch_size, subject_num, _, _ = source_data.shape
        
        total_loss = 0.0
        total_rec_loss = 0.0
        total_sim_loss = 0.0
        num_processed = 0

        # 🎯 핵심: subject_num 축 기준으로 피험자별 순환 처리 (training_step과 동일)
        for subject_idx in range(subject_num):
            try:
                # 현재 피험자의 데이터 추출 (축 기준 슬라이싱)
                subject_batch_data = source_data[:, subject_idx, :, :]  # (batch_size, time_steps, features)
                subject_correspondence = correspondence_data[subject_idx,:, :, :]  # (batch_size, subject_nums*time_steps, features)
                subject_mask = subject_mask_data[subject_idx, :, :]  # (batch_size, subject_num)

                # print(f"🔍 Validation batch shapes: source_data={subject_batch_data.shape}, correspondence_data={subject_correspondence.shape}")
                
                # 피험자 ID 텐서 생성
                current_subject_ids = torch.full(
                    (batch_size,), 
                    subject_idx, 
                    dtype=torch.long, 
                    device=source_data.device
                )
                # print(f"🔍 Current subject IDs: {current_subject_ids}")
                
                # Forward pass (validation에서는 GRL 없음: m=0.0)
                rec_loss, sim_loss = self.forward(
                    subject_batch_data, subject_correspondence, 
                    current_subject_ids, m=0.0, subject_mask=subject_mask
                )
                
                # Loss 계산
                subject_loss = rec_loss + self.beta * sim_loss
                
                # Loss 누적
                total_loss += subject_loss.detach()
                total_rec_loss += rec_loss.detach()
                total_sim_loss += sim_loss.detach()
                num_processed += 1
                
                # Debug info (첫 번째 배치의 첫 번째 피험자에서만)
                if batch_idx == 0 and subject_idx == 0:
                    print(f"🔍 Val Subject {subject_idx}: rec_loss={rec_loss:.4f}, sim_loss={sim_loss:.4f}")
                    print(f"🔍 Val Subject shapes: data={subject_batch_data.shape}, correspondence={subject_correspondence.shape}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"⚠️ Subject {subject_idx} validation failed: {e}") 
        
        # 평균 계산
        if num_processed > 0:
            avg_total_loss = total_loss / num_processed
            avg_rec_loss = total_rec_loss / num_processed
            avg_sim_loss = total_sim_loss / num_processed
        else:
            avg_total_loss = torch.tensor(0.0, device=self.device)
            avg_rec_loss = torch.tensor(0.0, device=self.device)
            avg_sim_loss = torch.tensor(0.0, device=self.device)
        
        # 메트릭 업데이트
        self.val_loss_metrics['total_loss'].update(
            avg_total_loss.item() if isinstance(avg_total_loss, torch.Tensor) else avg_total_loss
        )
        self.val_loss_metrics['rec_loss'].update(
            avg_rec_loss.item() if isinstance(avg_rec_loss, torch.Tensor) else avg_rec_loss
        )
        self.val_loss_metrics['sim_loss'].update(
            avg_sim_loss.item() if isinstance(avg_sim_loss, torch.Tensor) else avg_sim_loss
        )
        
        # 로깅
        self.log_dict({
            'val_total_loss': avg_total_loss,
            'val_rec_loss': avg_rec_loss,
            'val_sim_loss': avg_sim_loss,
            'val_subjects_processed': num_processed,
            'val_subject_num': subject_num,
            'val_batch_size': batch_size,
            # Placeholder metrics for Ray Tune compatibility (pretraining doesn't have classification)
            'val_macro_acc': 0.0,  # Will be replaced in fine-tuning phase
            'val_micro_acc': 0.0,  # Will be replaced in fine-tuning phase
            'val_acc': 0.0  # Required by Ray Tune AsyncHyperBandScheduler
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return avg_total_loss
    
    def _get_subject_val_batch(self, subject_id: str, subject_val_dataloader):
        """Subject별 validation 배치 가져오기 (training과 동일한 방식)"""
        
        # Validation iterator 관리
        val_iterator_key = f"{subject_id}_val"
        if val_iterator_key not in self.subject_iterators:
            self.subject_iterators[val_iterator_key] = iter(subject_val_dataloader)
        
        try:
            # Subject에서 validation 배치 가져오기
            batch = next(self.subject_iterators[val_iterator_key])
            return batch
        except StopIteration:
            # Iterator 재시작
            self.subject_iterators[val_iterator_key] = iter(subject_val_dataloader)
            try:
                batch = next(self.subject_iterators[val_iterator_key])
                return batch
            except StopIteration:
                return None
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """시각화를 위한 attention weights 추출."""
        with torch.no_grad():
            # attention layer에 접근
            attention_layer = self.model.attention_layer
            x_weighted = attention_layer(x, x.shape[0], self.time_steps)
            
            # attention weights 계산
            x_reshape = torch.reshape(x, [-1, self.input_dim])
            attn_weights = F.softmax(
                torch.mm(x_reshape, attention_layer.w_linear) + 
                attention_layer.u_linear, 
                dim=1
            )
            attn_weights = torch.reshape(
                attn_weights, [x.shape[0], self.time_steps, self.input_dim]
            )
        
        return attn_weights