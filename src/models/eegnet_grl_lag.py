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

from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
from src.models.base_eegnet import BaseEEGNetGRL

from typing import List, Callable, Dict, Optional

class LearnableLagSelection(nn.Module):
    def __init__(self, max_lag=32, num_lags=8, temperature=1.0):
        super().__init__()
        self.max_lag = max_lag
        self.num_lags = num_lags
        self.temperature = temperature
        self.lag_weights = nn.Parameter(torch.randn(max_lag))
        
    def forward(self, features):
        # 소프트 어텐션을 통한 지연 선택
        if self.training:
            # Training: Use Gumbel Softmax for differentiable sampling
            lag_probs = F.gumbel_softmax(self.lag_weights, tau=self.temperature, hard=False)
            top_k_values, top_k_indices = torch.topk(lag_probs, self.num_lags)
        else:
            # Inference: Use regular softmax and top-k
            lag_probs = F.softmax(self.lag_weights, dim=0)
            top_k_indices = torch.topk(lag_probs, self.num_lags).indices
        return top_k_indices

class EEGNetLNLLag(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3, lnl_lambda=0.01, lmi_mode='mean', 
                 lnl_lambda_learnable=False,
                 lag_strategy: str = 'learnable',      # 'fixed' | 'autocorr' | 'learnable' | ...
                 lag_interval: int = 4,           # used by 'fixed'
                 autocorr_factor: float = 1.0,    # used by 'autocorr'
                 max_lag: int = 32,               # used by 'learnable'
                 num_lags: int = 8,               # used by 'learnable'
                 lag_temperature: float = 1.0,    # used by 'learnable'
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None,
                 **kwargs):
        super(EEGNetLNLLag, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor, class_weight=class_weight)
        self.save_hyperparameters()

        # mean, max, None
        self.lmi_mode = lmi_mode
        self.current_lag_indices: torch.Tensor = None


        self.grl_lambda = grl_lambda
        # self.lnl_lambda = lnl_lambda
        if lnl_lambda_learnable:
            self.lnl_lambda = nn.Parameter(
                torch.full((nb_classes,), lnl_lambda, dtype=torch.float)
            )
        else:
            self.lnl_lambda = lnl_lambda
        
        # -------- Lag selection registry --------------------------------
        self.lag_strategy = lag_strategy.lower()
        self._laggers: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            'fixed': lambda feats: fixed_interval_lags(
                feats.shape[-1], interval=lag_interval),
            'autocorr': lambda feats: autocorr_topk_lags(
                feats, factor=autocorr_factor),
            'learnable': lambda feats: self.learnable_lag_selector(feats),
        }
        if self.lag_strategy not in self._laggers:
            raise ValueError(f"Unknown lag_strategy '{lag_strategy}'. "
                             f"Available: {list(self._laggers.keys())}")

        # Learnable lag selection module
        if self.lag_strategy == 'learnable':
            self.learnable_lag_selector = LearnableLagSelection(
                max_lag=max_lag, num_lags=num_lags, temperature=lag_temperature)
        else:
            self.learnable_lag_selector = None

        self.domain_criterion = nn.CrossEntropyLoss(reduction='none') if self.lmi_mode=='max' else nn.CrossEntropyLoss()

        self.domain_softmax = nn.Softmax(dim=-1)

        self.feature_extractor = EEGFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
        )

        # 타겟 분류기
        self.target_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), nb_classes, bias=True)
        )

        # 도메인 분류기
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), domain_classes, bias=True)
        )

        print(domain_classes)

        self.apply(self.weight_init)
    
    def configure_optimizers(self):
        # Base parameters for target and domain classifiers
        target_params = list(self.feature_extractor.parameters()) + list(self.target_classifier.parameters())
        domain_params = list(self.domain_classifier.parameters())
        
        # Add learnable lag selection parameters if using learnable strategy
        if self.lag_strategy == 'learnable' and self.learnable_lag_selector is not None:
            target_params += list(self.learnable_lag_selector.parameters())
        
        target_optimizer = torch.optim.Adam(target_params, lr=self.lr)
        domain_optimizer = torch.optim.Adam(domain_params, lr=self.lr)
        
        target_optimizer_scheduler = None
        domain_optimizer_scheduler = None

        if self.scheduler_name=='CosineAnnealingWarmUpRestarts':
            from src.utils.schedulers.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
            target_optimizer_scheduler = CosineAnnealingWarmUpRestarts(target_optimizer, T_0=50, 
                                                                T_mult=1.25, eta_max=0.1,  
                                                                T_up=10, gamma=0.75, last_epoch=-1)
            domain_optimizer_scheduler = CosineAnnealingWarmUpRestarts(domain_optimizer, T_0=50, 
                                                                T_mult=1.25, eta_max=0.1,  
                                                                T_up=10, gamma=0.75, last_epoch=-1)
        elif self.scheduler_name=='ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            target_optimizer_scheduler = ReduceLROnPlateau(target_optimizer,mode='max',factor=0.5, patience=5, threshold=0.001,cooldown=10)
            domain_optimizer_scheduler = ReduceLROnPlateau(domain_optimizer,mode='min',factor=0.5, patience=5, threshold=0.001,cooldown=10)
        else:
            print(">> Not use LR scheduler?")

        if target_optimizer_scheduler and domain_optimizer_scheduler:
            return [ 
            {'optimizer': target_optimizer, 
             'lr_scheduler': {'scheduler':target_optimizer_scheduler, 'name':'target_lr', 'monitor':self.target_monitor}},
            {'optimizer': domain_optimizer, 
             'lr_scheduler': {'scheduler':domain_optimizer_scheduler, 'name':'domain_lr', 'monitor':self.domain_monitor}}
            ]
        else:
            return [target_optimizer, domain_optimizer]
    
    def forward(self, x, second_step=False):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)        

        target_output = self.target_classifier(features)
        
        if second_step:
            features = self.grl(features)
            domain_output = self.domain_classifier(features)
        else:
            domain_output = None
        
        return target_output, domain_output, features

    def _select_lags(self, features: torch.Tensor) -> torch.Tensor:
        """Return 1-D tensor of lag indices according to strategy."""
        return self._laggers[self.lag_strategy](features)
    
    def _calc_lag_mi(self, features, target_labels):
        lags = self._select_lags(features)                     # (k,)
        self.current_lag_indices = lags                       # cache for 2nd step

        # Create lagged features using the roll function
        # (batch_size, Lagged_time, Chans, 1, Samples)
        lagged_features = torch.cat([torch.roll(features, -int(l), -1).unsqueeze(1)
                            for l in lags], dim=1)            # (B,k,C,1,S)

        # Flatten the features for the domain classifier
        batch_size, lagged_time, Chans, _, Samples = lagged_features.shape
        lagged_features = lagged_features.view(batch_size * lagged_time, Chans, 1, Samples)

        # For learnable lag selection, we need to handle gradient flow differently
        if self.lag_strategy == 'learnable':
            # Apply GRL to lagged features for learnable case
            lagged_features = self.grl(lagged_features)

        # (batch_size * Lagged_time, domain_classes)
        domain_output = self.domain_classifier(lagged_features)

        # Reshape back to (batch_size, Lagged_time, domain_classes)
        domain_output = domain_output.view(batch_size, lagged_time, -1)
        domain_output = self.domain_softmax(domain_output)

        epsilon = 1e-8
        # (batch_size, Lagged_time)
        lag_mi_values = torch.sum(domain_output * torch.log(domain_output+epsilon), dim=-1)

        if self.lmi_mode == 'mean':
            lag_mi_values = torch.mean(lag_mi_values, dim=1)
        elif self.lmi_mode == 'max':
            lag_mi_values, self.lag_mi_max_index = torch.max(lag_mi_values, dim=1)
        else:
            raise ValueError(f"Invalid lmi_mode: {self.lmi_mode}. Expected 'mean' or 'max'.")

        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            if target_labels is None:
                raise ValueError("target_labels must be provided when lnl_lambda is learnable.")
            lam_batch = self.lnl_lambda[target_labels]          # (B,)
            lag_mi_value = (lam_batch * lag_mi_values).mean()
        else:
            lag_mi_value = lag_mi_values.mean()

        return lag_mi_value
    
    def _calc_domain_loss(self, features, domain_labels):
        # (batch_size, Chans, 1, Samples)
        lags = self.current_lag_indices

        # Create lagged features using the roll function
        # (batch_size, Lagged_time, Chans, 1, Samples)
        lagged_features = torch.cat([torch.roll(features, -int(l), -1).unsqueeze(1)
                            for l in lags], dim=1)

        # Flatten the features for the domain classifier
        batch_size, lagged_time, Chans, _, Samples = lagged_features.shape
        lagged_features = lagged_features.view(batch_size * lagged_time, Chans, 1, Samples)

        # (batch_size * Lagged_time, domain_classes)
        domain_output = self.domain_classifier(lagged_features)

        # (batch_size,) -> (batch_size, lagged_time) -> (batch_size * lagged_time)
        expanded_labels = domain_labels.unsqueeze(1).expand(-1, lagged_time).reshape(-1)

        if self.lmi_mode == 'mean':
            domain_loss = self.domain_criterion(domain_output, expanded_labels)            
        elif self.lmi_mode == 'max':
            per_sample_loss = self.domain_criterion(domain_output, expanded_labels).view(batch_size, lagged_time)
            domain_loss = per_sample_loss[torch.arange(batch_size), self.lag_mi_max_index].mean()
        else:
            raise ValueError(f"Invalid lmi_mode: {self.lmi_mode}. Expected 'mean' or 'max'.")

        return domain_loss

    def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            return target_loss + loss_pseudo_pred_domain
        else:
            return target_loss + self.lnl_lambda*loss_pseudo_pred_domain

    def _first_step(self, x, target_labels, target_optimizer, domain_optimizer, prefix='train'):
        # first step
        target_output, _, features = self(x)
        target_loss = self.target_criterion(target_output, target_labels)
        # domain_output = self.domain_softmax(domain_output)

        # loss_pseudo_pred_domain = torch.mean(torch.sum(domain_output*torch.log(domain_output),1))
        loss_pseudo_pred_domain = self._calc_lag_mi(features, target_labels)
        first_step_total_loss = self._calc_first_step_loss(target_loss, loss_pseudo_pred_domain)        

        target_preds = torch.argmax(target_output, dim=1)

        # optim -> feature_extractor + target_classifier
        if prefix=='train':
            target_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            self.manual_backward(first_step_total_loss)
            target_optimizer.step()

        self.log(f'{prefix}_first_step_total_loss', first_step_total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_MI', loss_pseudo_pred_domain, on_epoch=True, prog_bar=True, sync_dist=True)

        getattr(self, f'{prefix}_target_metrics').update(target_preds, target_labels)
        self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())

        return target_labels, target_preds
    
    def _second_step(self, x, domain_labels, target_optimizer, domain_optimizer, prefix='train'):
        # second step
        _, domain_output, features = self(x, second_step=True)
        # domain_loss = self.domain_criterion(domain_output, domain_labels)
        domain_loss = self._calc_domain_loss(features, domain_labels)

        # optim only domain classifier
        if prefix=='train':
            target_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            self.manual_backward(domain_loss)
            target_optimizer.step()
            domain_optimizer.step()
                
        # 예측 및 정확도 계산
        domain_preds = torch.argmax(domain_output, dim=1)

        getattr(self, f'{prefix}_domain_metrics').update(domain_preds, domain_labels)

        self.log(f'{prefix}_domain_loss', domain_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return domain_labels, domain_preds

    def test_step(self, batch, batch_idx):
        x, target_labels = batch
        target_output, _, _ = self(x)

        target_loss = self.target_criterion(target_output, target_labels)
        target_preds = torch.argmax(target_output, dim=1)

        self.log('test_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        getattr(self, 'test_target_metrics').update(target_preds, target_labels)
        self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())

# -------- Lag-selection helpers ---------------------------------------------
def fixed_interval_lags(length: int, interval: int = 4, max_len: Optional[int] = None) -> torch.Tensor:
    """0,interval,2*interval,…  (length limited by max_len or series length)."""
    idx = torch.arange(0, length, interval)
    if max_len is not None:
        idx = idx[:max_len]
    return idx

import math
@torch.no_grad()
def autocorr_topk_lags(features: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Autoformer-style top-k lag extraction.
    features: (B, C, 1, S)
    returns : (k,) tensor of positive lag indices (0 excluded)
    """
    _, _, _, S = features.shape
    f = features.squeeze(2)                       # (B,C,S)
    fft = torch.fft.rfft(f, dim=-1)
    ac = torch.fft.irfft(fft * fft.conj(), n=S, dim=-1).mean(dim=(0, 1))  # (S,)
    ac[0] = -1e9                                  # ignore lag 0
    k = max(1, int(factor * math.log(S)))
    return torch.topk(ac, k, dim=-1).indices      # (k,)