import os
import sys

# Use project setup utilities
try:
    from src.utils.project_setup import project_paths
except ImportError:
    import os
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
from src.models.base_eegnet import BaseEEGNet

class GroupDROLossComputer(nn.Module):
    def __init__(self, criterion, n_groups, group_counts, step_size=0.01,
                 normalize_loss=False, is_robust=True, adj=None):
        super(GroupDROLossComputer, self).__init__()
        self.criterion = criterion
        self.n_groups = n_groups
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.is_robust = is_robust
        
        # --- 수정된 부분: register_buffer 사용 ---
        # register_buffer(버퍼이름, 텐서) 형태로 등록
        self.register_buffer('adv_probs', torch.ones(n_groups) / n_groups)
        
        # 그룹별 통계
        _group_counts = group_counts if torch.is_tensor(group_counts) else torch.tensor(group_counts)
        self.register_buffer('group_counts', _group_counts)
        self.register_buffer('group_frac', self.group_counts / self.group_counts.sum())
        # 오타 수정: grouop_list -> group_list
        self.register_buffer('group_list', torch.arange(self.n_groups).unsqueeze(1).long())

        # 조정값 (adjustment) 설정
        if adj is not None:
            _adj = torch.from_numpy(adj).float() if isinstance(adj, np.ndarray) else torch.tensor(adj).float()
        else:
            _adj = torch.zeros(self.n_groups).float()
        self.register_buffer('adj', _adj)
        
        # 지수 이동 평균을 위한 변수들
        self.register_buffer('exp_avg_loss', torch.zeros(n_groups))
        self.register_buffer('exp_avg_initialized', torch.zeros(n_groups, dtype=torch.uint8)) # byte 대신 dtype 사용
        self.gamma = 0.1
        
        self.init_stats()
    
    def compute_group_avg(self, losses, group_idx):
        """그룹별 평균 손실 계산"""
        # 각 그룹에 대한 마스크 생성
        group_map = (group_idx == self.group_list).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # 0으로 나누기 방지
        
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count
    
    def update_exp_avg_loss(self, group_loss, group_count):
        """지수 이동 평균 업데이트"""
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)
    
    def compute_robust_loss(self, group_loss, group_count):
        """Group DRO 손실 계산"""
        adjusted_loss = group_loss
        
        # adjustment 적용 (있는 경우)
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
            
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        
        # 그룹별 가중치 업데이트 (어려운 그룹에 더 높은 가중치)
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        
        # 가중 평균 손실
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs
    
    def loss(self, yhat, y, group_idx, is_training=False):
        """메인 손실 계산 함수"""
        # 샘플별 손실 계산
        per_sample_losses = self.criterion(yhat, y)
        
        # 그룹별 손실 및 정확도 계산
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count  = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)
        
        # 지수 이동 평균 업데이트
        self.update_exp_avg_loss(group_loss, group_count)
        
        # 최종 손실 계산
        if self.is_robust:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None
        
        # 통계 업데이트
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        
        return actual_loss
    
    def init_stats(self):
        """통계 초기화"""
        # --- 수정된 부분: register_buffer 사용 ---
        self.register_buffer('processed_data_counts', torch.zeros(self.n_groups))
        self.register_buffer('avg_group_loss', torch.zeros(self.n_groups))
        self.register_buffer('avg_group_acc', torch.zeros(self.n_groups))
        
        # 아래 값들은 텐서가 아니므로 그대로 두거나, 텐서로 관리하려면 동일하게 버퍼로 등록
        self.register_buffer('avg_actual_loss', torch.tensor(0.))
        self.register_buffer('avg_acc', torch.tensor(0.))
        self.register_buffer('batch_count', torch.tensor(0.))

    def reset_stats(self):
        """
        통계 초기화: 새로운 텐서를 생성하는 대신,
        기존 버퍼의 값을 제자리에서(in-place) 0으로 리셋합니다.
        """
        # self.processed_data_counts는 이미 GPU에 있으므로,
        # zero_()를 호출하면 GPU 상에서 값이 0으로 바뀝니다.
        self.processed_data_counts.zero_()
        
        # 다른 통계 버퍼들도 동일하게 처리합니다.
        self.avg_group_loss.zero_()
        self.avg_group_acc.zero_()
        self.avg_actual_loss.zero_()
        self.avg_acc.zero_()
        self.batch_count.zero_()
    
    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        """통계 업데이트"""
        # 그룹별 손실 및 정확도 업데이트
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc
        
        # 전체 손실 업데이트
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss
        
        # 카운트 업데이트
        self.processed_data_counts += group_count
        self.batch_count += 1
        
        # 전체 정확도 계산
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_acc = group_frac @ self.avg_group_acc
    
    def get_stats(self):
        """통계 반환"""
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'adv_prob_group:{idx}'] = self.adv_probs[idx].item()
        
        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()
        stats_dict['worst_group_acc'] = self.avg_group_acc.min().item()
        
        return stats_dict


class EEGNetDRO(BaseEEGNet):
    """Group DRO를 적용한 EEGNet 모델"""
    def __init__(self, nb_classes=2, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_worst_group_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None,
                 # Group DRO 관련 파라미터
                 n_groups=None, group_counts=None, step_size=0.01, adj=None,
                 normalize_loss=False, is_robust=True,
                 **kwargs):
        super(EEGNetDRO, self).__init__(nb_classes=nb_classes, norm_rate=norm_rate, lr=lr, 
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, class_weight=class_weight)
        self.save_hyperparameters()
        
        # Group DRO 설정
        self.n_groups = n_groups
        self.is_robust = is_robust
        
        # 기본 criterion 설정
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Group DRO Loss Computer 초기화
        self.train_loss_computer = GroupDROLossComputer(
            criterion=self.base_criterion,
            n_groups=n_groups,
            group_counts=group_counts['train'],
            step_size=step_size,
            normalize_loss=normalize_loss,
            is_robust=is_robust,
            adj=adj
        )

        # val 시에도 동일한 Loss Computer 사용
        self.val_loss_computer = GroupDROLossComputer(
            criterion=self.base_criterion,
            n_groups=n_groups,
            group_counts=group_counts['val'],
            step_size=step_size,
            is_robust=is_robust,
        )

        # 모델 구조
        self.feature_extractor = EEGFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), nb_classes, bias=True)
        )
        
        self.apply(self.weight_init)
    
    def forward(self, x):
        """순전파"""
        # 입력 형태 변환
        # (B, C, S, 1) -> (B, 1, C, S)
        x = x.permute(0, 3, 1, 2)
        
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

    def _step_func(self, batch, prefix='train'):
        x, labels = batch
        # return x, (y, domain_label, group_label)
        # labels가 y, domain_y, group_label를 포함하지 않는 다면 error 발생
        print(f"labels: {labels}")
        if prefix == 'test':
            labels = labels
        else:
            labels, domain_labels, group_labels = labels
        
        output = self(x)

        loss = self.criterion(output, labels)

        # Group DRO 손실 계산
        if prefix != 'test':
            loss_computer = getattr(self, f"{prefix}_loss_computer")
            DRO_loss = loss_computer.loss(output, labels, group_labels, is_training=(prefix == 'train'))
        else:
            DRO_loss = loss

        preds = torch.argmax(output, dim=1)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if prefix != 'test':
            self.log(f"{prefix}_DRO_loss", DRO_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        getattr(self, f'{prefix}_metrics').update(preds, labels)

        return DRO_loss, labels, preds
    
    def _log_DRO_stats(self, prefix='train'):
        """Group DRO 통계 로그"""
        stats_computer = getattr(self, f"{prefix}_loss_computer")
        stats = stats_computer.get_stats()
        for key, value in stats.items():
            self.log(f"{prefix}_{key}", value, on_epoch=True, sync_dist=True)
    
    def _epoch_end_func(self, prefix='train'):
        super()._epoch_end_func(prefix)
        if prefix != 'test':
            # Group DRO 통계 로그
            self._log_DRO_stats(prefix=prefix)
        if prefix == 'train':
            self.train_loss_computer.reset_stats()