"""
DMMR Fine-tuning Lightning Module.

Implements fine-tuning phase for emotion/task classification using pretrained representations.
"""
from typing import Dict, Any, Optional, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from .lightning_base import DMMRBaseLightningModule
from ..models.finetuning_model import DMMRFineTuningModel


class DMMRFineTuningModule(DMMRBaseLightningModule):
    """
    DMMR Fine-tuning Lightning Module.
    
    DMMR 파인튜닝 Lightning 모듈로 사전훈련된 attention과 encoder 가중치를 상속받고,
    새로운 감정 분류기를 추가하여 감정 인식 작업을 위한 파인tuning을 수행합니다.
    
    Args:
        pretrained_module: 사전훈련된 모듈 (선택적)
        pretrained_checkpoint_path: 사전훈련된 체크포인트 경로 (선택적)
        freeze_pretrained: 사전훈련 컴포넌트 동결 여부 (default: True)
        classifier_hidden_dim: 분류기 히든 차원 (default: 64)
        dropout_rate: 드롭아웃 비율 (default: 0.0)
        number_of_source: 소스 도메인 수 (default: 14)
    """
    
    def __init__(
        self,
        pretrained_module=None,
        pretrained_checkpoint_path: Optional[str] = None,
        freeze_pretrained: bool = True,
        classifier_hidden_dim: int = 64,
        dropout_rate: float = 0.0,
        number_of_source: int = 14,
        **kwargs
    ):
        # DMMR 파인튜닝 기본값
        dmmr_defaults = {
            'num_classes': 3,
            'input_dim': 310,
            'hidden_dim': 64,
            'batch_size': 10,
            'time_steps': 15,
            'learning_rate': 1e-4,  # 파인튜닝을 위한 낮은 학습률
        }
        
        # 파인튜닝 특화 파라미터를 먼저 설정
        self.freeze_pretrained = freeze_pretrained
        self.classifier_hidden_dim = classifier_hidden_dim
        # Handle both dropoutRate (from config) and dropout_rate (parameter)
        self.dropout_rate = kwargs.get('dropoutRate', dropout_rate)
        self.number_of_source = number_of_source
        
        config = {**dmmr_defaults, **kwargs}
        super().__init__(**config)
        
        # 사전훈련 가중치 로드
        if pretrained_module is not None:
            self._inherit_pretrained_weights(pretrained_module)
        elif pretrained_checkpoint_path is not None:
            self._load_pretrained_checkpoint(pretrained_checkpoint_path)
        
        # 🔧 원본 DMMR 방식: freeze_pretrained는 무조건 False로 처리 (원본에서는 freeze 안함)
        # 원본 DMMR에서는 모든 파라미터가 fine-tuning 단계에서 학습됨
        if self.freeze_pretrained:
            print("⚠️  DMMR Warning: freeze_pretrained=True but original DMMR doesn't freeze. Setting to False.")
            self.freeze_pretrained = False
            
        # 모든 파라미터의 gradient 활성화 보장 (원본 DMMR 방식)
        self._ensure_gradients_enabled()
        
        # 🔧 파라미터 gradient 상태 진단
        self._diagnose_gradient_status()

        self._verify_gradient_integrity()

        self.automatic_optimization = False
    
    def _build_model(self) -> None:
        """DMMR 파인튜닝 모델 구조 생성."""
        self.model = DMMRFineTuningModel(
            base_model=None,  # 나중에 사전훈련 가중치로 업데이트
            number_of_source=self.number_of_source,
            number_of_category=self.num_classes,
            batch_size=self.batch_size,
            time_steps=self.time_steps,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=1,
            dropout_rate=self.dropout_rate,
            num_classes=self.num_classes
        )
    
    def _inherit_pretrained_weights(self, pretrained_module) -> None:
        """사전훈련 모듈로부터 가중치 상속."""
        # 🔧 핵심 수정: state_dict 대신 직접 parameter 복사로 gradient graph 유지
        # Attention layer 가중치 복사
        pretrained_attention_state = pretrained_module.model.attention_layer.state_dict()
        finetuning_attention_state = self.model.attention_layer.state_dict()
        
        for key in pretrained_attention_state:
            if key in finetuning_attention_state:
                # 🎯 핵심: requires_grad 상태를 유지하며 parameter data만 복사
                with torch.no_grad():
                    finetuning_attention_state[key].copy_(pretrained_attention_state[key])
        
        # Shared encoder 가중치 복사  
        pretrained_encoder_state = pretrained_module.model.shared_encoder.state_dict()
        finetuning_encoder_state = self.model.shared_encoder.state_dict()
        
        for key in pretrained_encoder_state:
            if key in finetuning_encoder_state:
                # 🎯 핵심: requires_grad 상태를 유지하며 parameter data만 복사
                with torch.no_grad():
                    finetuning_encoder_state[key].copy_(pretrained_encoder_state[key])
        
        print("✅ Pre-trained weights successfully inherited with gradient preservation!")
    
    def _load_pretrained_checkpoint(self, checkpoint_path: str) -> None:
        """체크포인트로부터 사전훈련 가중치 로드."""
        from .pretraining import DMMRPreTrainingModule
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 임시 사전훈련 모듈 생성 및 상태 로드
        temp_module = DMMRPreTrainingModule()
        temp_module.load_state_dict(checkpoint['state_dict'])
        
        # 가중치 상속
        self._inherit_pretrained_weights(temp_module)
        
        print(f"✅ Pre-trained weights loaded from {checkpoint_path}")
    
    def _verify_gradient_integrity(self) -> None:
        """Weight transfer 후 gradient 무결성 검증"""
        problematic_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                problematic_params.append(name)
        
        if problematic_params:
            print(f"⚠️ Parameters without gradient: {problematic_params}")
            # 자동 복구 시도
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            print("🔧 Gradient requirements automatically restored")
    
    def _freeze_pretrained_components(self) -> None:
        """사전훈련 컴포넌트 동결."""
        frozen_params = 0
        for param in self.model.attention_layer.parameters():
            param.requires_grad = False
            frozen_params += 1
        
        for param in self.model.shared_encoder.parameters():
            param.requires_grad = False
            frozen_params += 1
        
        print(f"🔒 Pre-trained components frozen! ({frozen_params} parameters)")
    
    def _unfreeze_pretrained_components(self) -> None:
        """End-to-end 파인튜닝을 위한 사전훈련 컴포넌트 해동."""
        unfrozen_params = 0
        for param in self.model.attention_layer.parameters():
            param.requires_grad = True
            unfrozen_params += 1
        
        for param in self.model.shared_encoder.parameters():
            param.requires_grad = True
            unfrozen_params += 1
        
        print(f"🔓 Pre-trained components unfrozen! ({unfrozen_params} parameters)")
    
    def _ensure_gradients_enabled(self) -> None:
        """모든 파라미터의 gradient 활성화를 보장."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("🎯 All parameters gradient enabled!")
    
    def _diagnose_gradient_status(self) -> None:
        """파라미터 gradient 상태 진단."""
        attention_grads = sum(1 for p in self.model.attention_layer.parameters() if p.requires_grad)
        encoder_grads = sum(1 for p in self.model.shared_encoder.parameters() if p.requires_grad)  
        classifier_grads = sum(1 for p in self.model.cls_fc.parameters() if p.requires_grad)
        
        print(f"🔍 Gradient Status - Attention: {attention_grads}, Encoder: {encoder_grads}, Classifier: {classifier_grads}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Fine-tuning forward pass.
        
        Args:
            x: Input EEG data
            return_features: 중간 특징 반환 여부
            
        Returns:
            Classification logits 또는 (logits, features)
        """
        # 모델의 forward pass (분류 손실 없이 로짓만)
        x_pred, x_logits, _ = self.model.forward(x, None)
        
        if return_features:
            # 특징 추출을 위해 인코더까지만 실행
            x_att = self.model.attention_layer(x, x.shape[0], self.time_steps)
            features, _, _ = self.model.shared_encoder(x_att)
            return x_logits, features
        else:
            return x_logits
    
    def training_step(self, batch, batch_idx):
        """DMMR 원본 방식: 4D 데이터에서 subject_num 축 기준 피험자별 순환 Fine-tuning (Manual Optimization)"""
        optimizer = self.optimizers()
        
        source_batch_data, _, source_labels, _ = batch
        batch_size, _, _ = source_batch_data.shape

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        try:
            # Forward pass (분류용 모델만 사용)
            logits = self.forward(source_batch_data)

            # 분류 손실 계산
            subject_loss = F.cross_entropy(
                logits, 
                source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            )
            
            # 🔧 핵심: Manual backward (원본 DMMR Fine-tuning 방식 - 피험자별 독립적 backward)
            self.manual_backward(subject_loss)
            
            # Loss 및 정확도 누적
            total_loss += subject_loss.detach()
            
            # 예측값 계산 및 정확도 누적
            preds = torch.argmax(logits, dim=1)
            actual_labels = source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            total_correct += (preds == actual_labels).sum().item()
            total_samples += actual_labels.size(0)
            
            # Debug info (첫 번째 배치에서만)
            if batch_idx == 0:
                print(f"🎯 FineTune loss={subject_loss:.4f}")
                print(f"🔍 batch shapes: data={source_batch_data.shape}, labels={source_labels.shape}")

        except Exception as e:
            raise e
        
        # Optimizer step (모든 피험자 처리 후 - 원본 DMMR 방식)
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss = total_loss
        accuracy = total_correct
        
        # 메트릭 업데이트 (전체 배치 기준)
        self.train_metrics.update(preds, source_labels)
        
        # 로깅
        self.log_dict({
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'batch_size': batch_size
        }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        """DMMR 원본 방식: 4D 데이터에서 subject_num 축 기준 피험자별 순환 Fine-tuning Validation"""
        
        source_batch_data, _, source_labels, _ = batch
        batch_size, _, _ = source_batch_data.shape
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        try:
            # Forward pass
            logits = self.forward(source_batch_data)
            
            # 분류 손실 계산
            subject_loss = F.cross_entropy(
                logits,
                source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            )
            
            # Loss 및 정확도 누적
            total_loss += subject_loss.detach()
            
            # 예측값 계산 및 정확도 누적
            preds = torch.argmax(logits, dim=1)
            actual_labels = source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            total_correct += (preds == actual_labels).sum().item()
            total_samples += actual_labels.size(0)
            
            # Debug info (첫 번째 배치에서만)
            if batch_idx == 0:
                print(f"🎯 FineTune loss={subject_loss:.4f}")
                print(f"🔍 batch shapes: data={source_batch_data.shape}, labels={source_labels.shape}")
            
        except Exception as e:
            raise e
        
        avg_loss = total_loss
        accuracy = total_correct
        
        # 메트릭 업데이트 (전체 배치 기준)
        self.val_metrics.update(preds, source_labels)

        # 로깅
        self.log_dict({
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_batch_size': batch_size
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return avg_loss
    
    def test_step(self, batch, batch_idx):
        """DMMR 전용 Test step - validation_step과 독립적으로 구현"""
        
        source_batch_data, _, source_labels, _ = batch
        batch_size, _, _ = source_batch_data.shape
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        try:
            # Forward pass (test는 gradient 계산 없이)
            with torch.no_grad():
                logits = self.forward(source_batch_data)

            # 분류 손실 계산
            subject_loss = F.cross_entropy(
                logits, 
                source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            )
            
            # Loss 및 정확도 누적
            total_loss += subject_loss.detach()
            
            # 예측값 계산 및 정확도 누적
            preds = torch.argmax(logits, dim=1)
            actual_labels = source_labels.squeeze() if source_labels.dim() > 1 else source_labels
            total_correct += (preds == actual_labels).sum().item()
            total_samples += actual_labels.size(0)
            
            # 전체 예측값과 라벨 저장 (메트릭 계산용)
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_labels.extend(actual_labels.cpu().numpy().tolist())
            
            # Debug info (첫 번째 배치에서만)
            if batch_idx == 0:
                print(f"🎯 FineTune loss={subject_loss:.4f}")
                print(f"🔍 batch shapes: data={source_batch_data.shape}, labels={source_labels.shape}")
            
        except Exception as e:
            raise e
        
        avg_loss = total_loss
        accuracy = total_correct
        
        # 메트릭 업데이트 (test 전용)
        # Convert to tensors for metric calculation
        all_predictions_tensor = torch.tensor(all_predictions, device=self.device)
        all_labels_tensor = torch.tensor(all_labels, device=self.device)
        
        # Test metrics update
        self.test_metrics.update(all_predictions_tensor, all_labels_tensor)
        
        # 로깅 (test 전용)
        self.log_dict({
            'test_loss': avg_loss,
            'test_acc': accuracy,
            'test_batch_size': batch_size
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'test_loss': avg_loss,
            'test_acc': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
        }
    
    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of test epoch."""
        if hasattr(self, 'test_metrics'):
            # Compute metrics from metric collection
            test_metrics = self.test_metrics.compute()
            
            # Log all test metrics
            self.log_dict(test_metrics, prog_bar=True, sync_dist=True)
            
            # Reset metrics
            self.test_metrics.reset()
            
            print(f"🧪 Test completed. Final metrics: {test_metrics}")
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        # Compute metrics from metric collection
        val_metrics = self.val_metrics.compute()
        
        # Log all other metrics
        self.log_dict(val_metrics, prog_bar=False, sync_dist=True)
        
        # Reset metrics for next epoch
        self.val_metrics.reset()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        # Forward pass
        logits = self.forward(x)
        
        # 확률과 예측 클래스 반환
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': logits
        }
    
    def freeze_pretrained_layers(self):
        """사전훈련 레이어 동결을 위한 공개 메서드."""
        self._freeze_pretrained_components()
    
    def unfreeze_pretrained_layers(self):
        """End-to-end 훈련을 위한 사전훈련 레이어 해동."""
        self._unfreeze_pretrained_components()
    
    def get_feature_representations(self, x: torch.Tensor) -> torch.Tensor:
        """인코더로부터 특징 표현 추출."""
        with torch.no_grad():
            x_att = self.model.attention_layer(x, x.shape[0], self.time_steps)
            features, _, _ = self.model.shared_encoder(x_att)
        
        return features