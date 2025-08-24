import os
import sys
import torch
import torch.nn as nn
import math

from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
from src.models.base_eegnet import BaseEEGNetGRL

from typing import List, Callable, Dict, Optional

class EEGNetGRL(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3,
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None):
        super(EEGNetGRL, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor, class_weight=class_weight)
        self.save_hyperparameters()

        self.grl_lambda = grl_lambda

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

        self.apply(self.weight_init)

    def forward(self, x, second_step=False):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)

        target_output = self.target_classifier(features)
        
        if second_step:
            features = self.grl(features)
        domain_output = self.domain_classifier(features)
        
        return target_output, domain_output


class EEGNetLNL(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3, lnl_lambda=0.01,
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None, lnl_lambda_learnable=False,
                 channel_norm=True,
                 **kwargs):
        super(EEGNetLNL, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor, channel_norm=channel_norm,
                                        class_weight=class_weight)
        self.save_hyperparameters()

        self.grl_lambda = grl_lambda
        # self.lnl_lambda = lnl_lambda
        self.lnl_lambda_learnable = lnl_lambda_learnable
        if self.lnl_lambda_learnable:
            self.lnl_lambda = nn.Parameter(
                torch.full((nb_classes,), lnl_lambda, dtype=torch.float)
            )
        else:
            self.lnl_lambda = lnl_lambda

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

        self.apply(self.weight_init)
    
    def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            return target_loss + loss_pseudo_pred_domain
        else:
            return target_loss + self.lnl_lambda*loss_pseudo_pred_domain

    def forward(self, x, second_step=False):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        if self.channel_norm_layer is not None:
            x = self.channel_norm_layer(x)

        features = self.feature_extractor(x)

        target_output = self.target_classifier(features)
        
        if second_step:
            features = self.grl(features)
        domain_output = self.domain_classifier(features)
        
        return target_output, domain_output


class EEGNetMI(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3, lnl_lambda=0.01,
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None):
        super(EEGNetMI, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor, class_weight=class_weight)
        self.save_hyperparameters()

        self.grl_lambda = grl_lambda
        self.lnl_lambda = lnl_lambda

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

        self.apply(self.weight_init)
    
    def forward(self, x):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)

        target_output = self.target_classifier(features)
        domain_output = self.domain_classifier(features)
        
        return target_output, domain_output

    def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
        return target_loss + self.lnl_lambda*loss_pseudo_pred_domain

    def _first_step(self, x, target_labels, target_optimizer, domain_optimizer, prefix='train'):
        # first step
        target_output, domain_output = self(x)

        target_loss = self.target_criterion(target_output, target_labels)
        domain_loss = self.domain_criterion(domain_output, domain_labels)

        domain_output = self.domain_softmax(domain_output)


        # loss_pseudo_pred_domain = torch.mean(torch.sum(domain_output*torch.log(domain_output),1))
        loss_pseudo_pred_domain = self._calc_loss_pseudo_pred_domain(domain_output, target_labels)
        first_step_total_loss = self._calc_first_step_loss(target_loss, loss_pseudo_pred_domain)

        target_preds = torch.argmax(target_output, dim=1)
        domain_preds = torch.argmax(domain_output, dim=1)

        # optim -> feature_extractor + target_classifier
        if prefix=='train':
            target_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            self.manual_backward(first_step_total_loss)
            target_optimizer.step()
            domain_optimizer.step()

        self.log(f'{prefix}_first_step_total_loss', first_step_total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_MI', loss_pseudo_pred_domain, on_epoch=True, prog_bar=True, sync_dist=True)

        getattr(self, f'{prefix}_target_metrics').update(target_preds, target_labels)
        getattr(self, f'{prefix}_domain_metrics').update(domain_preds, domain_labels)

        self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())
        self.log(f'{prefix}_domain_loss', domain_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def _step_func(self, batch, prefix='train'):
        target_optimizer, domain_optimizer = self.optimizers()

        x, (target_labels, domain_labels) = batch  # 배치는 (데이터, 타겟 레이블, 도메인 레이블) 형태

        # first step
        self._first_step(x, target_labels, target_optimizer, domain_optimizer, prefix)
    

# class EEGNetLNLLag(BaseEEGNetGRL):
#     def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
#                  scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
#                  Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
#                  grl_lambda=0.3, lnl_lambda=0.01, lmi_mode='mean', lnl_lambda_learnable=True,
#                  dropoutType='Dropout', dropoutRate=0.5, class_weight=None,
#                  **kwargs):
#         super(EEGNetLNLLag, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
#                                         scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor)
#         self.save_hyperparameters()

#         # mean, max, None
#         self.lmi_mode = lmi_mode
#         self.lag_mi_max_index = None

#         self.grl_lambda = grl_lambda
#         # self.lnl_lambda = lnl_lambda
#         if lnl_lambda_learnable:
#             self.lnl_lambda = nn.Parameter(
#                 torch.full((nb_classes,), lnl_lambda, dtype=torch.float)
#             )
#         else:
#             self.lnl_lambda = lnl_lambda

#         if class_weight is not None:
#             weights = torch.tensor(class_weight, dtype=torch.float)
#         else:
#             weights = None

#         self.target_criterion = nn.CrossEntropyLoss(weight=weights)
#         self.domain_criterion = nn.CrossEntropyLoss(reduction='none') if self.lmi_mode=='max' else nn.CrossEntropyLoss()

#         self.domain_softmax = nn.Softmax(dim=-1)

#         self.feature_extractor = EEGFeatureExtractor(
#             Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
#             kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
#         )

#         # 타겟 분류기
#         self.target_classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(F2 * (Samples // 32), nb_classes, bias=True)
#         )

#         # 도메인 분류기
#         self.domain_classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(F2 * (Samples // 32), domain_classes, bias=True)
#         )

#         print(domain_classes)

#         self.apply(self.weight_init)
    
#     def forward(self, x, second_step=False):
#         # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
#         x = x.permute(0, 3, 1, 2)
#         features = self.feature_extractor(x)        

#         target_output = self.target_classifier(features)
        
#         if second_step:
#             features = self.grl(features)
#             domain_output = self.domain_classifier(features)
#         else:
#             domain_output = None
        
#         return target_output, domain_output, features
    
#     def _calc_lag_mi(self, features, target_labels):
#         # (batch_size, Chans, 1, Samples)
#         lagging_interval = 4  # Define the lagging time

#         # Create lagged features using the roll function
#         # (batch_size, Lagged_time, Chans, 1, Samples)
#         lagged_features = torch.cat([torch.roll(features, shifts=i, dims=-1).unsqueeze(1) 
#                                      for i in range(0, features.shape[-1], lagging_interval)], dim=1)

#         # Flatten the features for the domain classifier
#         batch_size, lagged_time, Chans, _, Samples = lagged_features.shape
#         lagged_features = lagged_features.view(batch_size * lagged_time, Chans, 1, Samples)

#         # (batch_size * Lagged_time, domain_classes)
#         domain_output = self.domain_classifier(lagged_features)

#         # Reshape back to (batch_size, Lagged_time, domain_classes)
#         domain_output = domain_output.view(batch_size, lagged_time, -1)

#         domain_output = self.domain_softmax(domain_output)

#         epsilon = 1e-8
#         # (batch_size, Lagged_time)
#         lag_mi_values = torch.sum(domain_output * torch.log(domain_output+epsilon), dim=-1)

#         if self.lmi_mode == 'mean':
#             lag_mi_values = torch.mean(lag_mi_values, dim=1)
#         elif self.lmi_mode == 'max':
#             lag_mi_values, self.lag_mi_max_index = torch.max(lag_mi_values, dim=1)
#         else:
#             raise ValueError(f"Invalid lmi_mode: {self.lmi_mode}. Expected 'mean' or 'max'.")

#         if isinstance(self.lnl_lambda, torch.nn.Parameter):
#             if target_labels is None:
#                 raise ValueError("target_labels must be provided when lnl_lambda is learnable.")
#             lam_batch = self.lnl_lambda[target_labels]          # (B,)
#             lag_mi_value = (lam_batch * lag_mi_values).mean()
#         else:
#             lag_mi_value = lag_mi_values.mean()

#         return lag_mi_value
    
#     def _calc_domain_loss(self, features, domain_labels):
#         # (batch_size, Chans, 1, Samples)
#         lagging_interval = 4  # Define the lagging time

#         # Create lagged features using the roll function
#         # (batch_size, Lagged_time, Chans, 1, Samples)
#         lagged_features = torch.cat([torch.roll(features, shifts=i, dims=-1).unsqueeze(1) 
#                                      for i in range(lagging_interval, features.shape[-1], lagging_interval)], dim=1)

#         # Flatten the features for the domain classifier
#         batch_size, lagged_time, Chans, _, Samples = lagged_features.shape
#         lagged_features = lagged_features.view(batch_size * lagged_time, Chans, 1, Samples)

#         # (batch_size * Lagged_time, domain_classes)
#         domain_output = self.domain_classifier(lagged_features)

#         # (batch_size,) -> (batch_size, lagged_time) -> (batch_size * lagged_time)
#         expanded_labels = domain_labels.unsqueeze(1).expand(-1, lagged_time).reshape(-1)

#         if self.lmi_mode == 'mean':
#             domain_loss = self.domain_criterion(domain_output, expanded_labels)            
#         elif self.lmi_mode == 'max':
#             print('★'*50)
#             print(expanded_labels.shape)
#             print()
#             print(domain_output.shape)
#             per_sample_loss = self.domain_criterion(domain_output, expanded_labels)
#             print(per_sample_loss.unique())
#             print('★'*50)
#             per_sample_loss = per_sample_loss.view(batch_size, lagged_time)
#             print(per_sample_loss)
#             print('★'*50)
#             selected_loss = per_sample_loss[torch.arange(batch_size), self.lag_mi_max_index]
#             print(selected_loss)
#             print('★'*50)
#             domain_loss = torch.mean(selected_loss)
#             print('★'*50)
#             print(domain_loss)
#         else:
#             raise ValueError(f"Invalid lmi_mode: {self.lmi_mode}. Expected 'mean' or 'max'.")

#         return domain_loss

#     def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
#         if isinstance(self.lnl_lambda, torch.nn.Parameter):
#             return target_loss + loss_pseudo_pred_domain
#         else:
#             return target_loss + self.lnl_lambda*loss_pseudo_pred_domain

#     def _first_step(self, x, target_labels, target_optimizer, domain_optimizer, prefix='train'):
#         # first step
#         target_output, _, features = self(x)
#         target_loss = self.target_criterion(target_output, target_labels)
#         # domain_output = self.domain_softmax(domain_output)

#         # loss_pseudo_pred_domain = torch.mean(torch.sum(domain_output*torch.log(domain_output),1))
#         loss_pseudo_pred_domain = self._calc_lag_mi(features, target_labels)
#         first_step_total_loss = self._calc_first_step_loss(target_loss, loss_pseudo_pred_domain)        

#         target_preds = torch.argmax(target_output, dim=1)

#         # optim -> feature_extractor + target_classifier
#         if prefix=='train':
#             target_optimizer.zero_grad()
#             domain_optimizer.zero_grad()
#             self.manual_backward(first_step_total_loss)
#             target_optimizer.step()

#         self.log(f'{prefix}_first_step_total_loss', first_step_total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
#         self.log(f'{prefix}_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)
#         self.log(f'{prefix}_MI', loss_pseudo_pred_domain, on_epoch=True, prog_bar=True, sync_dist=True)

#         getattr(self, f'{prefix}_target_metrics').update(target_preds, target_labels)
#         self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())

#         return target_labels, target_preds
    
#     def _second_step(self, x, domain_labels, target_optimizer, domain_optimizer, prefix='train'):
#         # second step
#         _, domain_output, features = self(x, second_step=True)
#         # domain_loss = self.domain_criterion(domain_output, domain_labels)
#         domain_loss = self._calc_domain_loss(features, domain_labels)

#         # optim only domain classifier
#         if prefix=='train':
#             target_optimizer.zero_grad()
#             domain_optimizer.zero_grad()
#             self.manual_backward(domain_loss)
#             target_optimizer.step()
#             domain_optimizer.step()
                
#         # 예측 및 정확도 계산
#         domain_preds = torch.argmax(domain_output, dim=1)

#         getattr(self, f'{prefix}_domain_metrics').update(domain_preds, domain_labels)

#         self.log(f'{prefix}_domain_loss', domain_loss, on_epoch=True, prog_bar=True, sync_dist=True)

#         return domain_labels, domain_preds

#     def test_step(self, batch, batch_idx):
#         x, target_labels = batch
#         target_output, _, _ = self(x)

#         target_loss = self.target_criterion(target_output, target_labels)
#         target_preds = torch.argmax(target_output, dim=1)

#         self.log('test_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)

#         getattr(self, 'test_target_metrics').update(target_preds, target_labels)
#         self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())


class EEGNetLNLLag(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3, lnl_lambda=0.01, lmi_mode='mean', 
                 lnl_lambda_learnable=True,
                 lag_strategy: str = 'autocorr',      # 'fixed' | 'autocorr' | ...
                 lag_interval: int = 4,           # used by 'fixed'
                 autocorr_factor: float = 1.0,    # used by 'autocorr'
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None,
                 **kwargs):
        super(EEGNetLNLLag, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor)
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
        }
        if self.lag_strategy not in self._laggers:
            raise ValueError(f"Unknown lag_strategy '{lag_strategy}'. "
                             f"Available: {list(self._laggers.keys())}")

        if class_weight is not None:
            weights = torch.tensor(class_weight, dtype=torch.float)
        else:
            weights = None

        self.target_criterion = nn.CrossEntropyLoss(weight=weights)
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


from src.models.components.auto_correlation import AutoCorrelationLayer, AutoCorrelation
class EEGNetLNLAutoCorrelation(BaseEEGNetGRL):
    def __init__(self,nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, 
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16, 
                 grl_lambda=0.3, lnl_lambda=0.01, lnl_lambda_learnable=True,
                 dropoutType='Dropout', dropoutRate=0.5, class_weight=None, 
                 **kwargs):
        super(EEGNetLNLAutoCorrelation, self).__init__(nb_classes=nb_classes, domain_classes=domain_classes, norm_rate=norm_rate, lr=lr, grl_lambda=grl_lambda,
                                        scheduler_name=scheduler_name, target_monitor=target_monitor, domain_monitor=domain_monitor)
        self.save_hyperparameters()

        # self.lnl_lambda = lnl_lambda
        self.lnl_lambda_learnable = lnl_lambda_learnable
        if self.lnl_lambda_learnable:
            # learnable lambda
            self.lnl_lambda = nn.Parameter(
                torch.full((nb_classes,), lnl_lambda, dtype=torch.float)
            )
        else:
            self.lnl_lambda = lnl_lambda
        self.grl_lambda = grl_lambda


        if class_weight is not None:
            weights = torch.tensor(class_weight, dtype=torch.float)
        else:
            weights = None

        self.target_criterion = nn.CrossEntropyLoss(weight=weights)
        self.domain_criterion = nn.CrossEntropyLoss()

        self.domain_softmax = nn.Softmax(dim=-1)

        self.feature_extractor = EEGFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType
        )

        self.auto_correlation = AutoCorrelationLayer(
            AutoCorrelation(False, 1.0, attention_dropout=dropoutRate, output_attention=True),
            F2, 4
        )
        self.dropout = nn.Dropout(dropoutRate)
    
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

        self.apply(self.weight_init)
    
    def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            return target_loss + loss_pseudo_pred_domain
        else:
            return target_loss + self.lnl_lambda*loss_pseudo_pred_domain

    def forward(self, x, second_step=False):
        # 입력 형태 변환: (batch_size, Chans, Samples, 1) -> (batch_size, 1, Chans, Samples)
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)

        # print emoge
        # print("***"*30)
        # print("Features shape before auto-correlation:")
        # print(features.shape)
        # b, c, 1, s -> b, s, c
        # s== length of time series, c== number of channels
        features = features.permute(0, 3, 1, 2).squeeze(3)
        # print("Features shape after shape chagne and before auto-correlation:")
        # print(features.shape)
        # Auto-correlation
        features, attn = self.auto_correlation(features, features, features, attn_mask=None)
        # features = features + self.dropout(features)
        features = self.dropout(features)
        # print("Features shape after auto-correlation:")
        # print(features.shape)
        # print("***"*30)

        target_output = self.target_classifier(features)
        
        if second_step:
            features = self.grl(features)
        domain_output = self.domain_classifier(features)
        
        return target_output, domain_output
    
    def apply_max_norm(self):
        # classifier가 반드시 두 번째 레이어(nn.Linear)를 포함해야 함
        with torch.no_grad():
            weight = self.target_classifier[1].weight
            norm = weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            weight.data *= desired / (1e-6 + norm)

            if self.auto_correlation is not None:
                for param in self.auto_correlation.parameters():
                    norm = param.data.norm(2)
                    desired = torch.clamp(norm, max=self.norm_rate)
                    param.data *= desired / (1e-6 + norm)


            if self.lnl_lambda_learnable:
                norm = self.lnl_lambda.data.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=self.norm_rate)
                self.lnl_lambda.data *= desired / (1e-6 + norm)
    
    def configure_optimizers(self):
        from src.utils.schedulers.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from torch.optim import AdamW

        target_params = list(self.feature_extractor.parameters()) + \
                    list(self.target_classifier.parameters()) + \
                    list(self.auto_correlation.parameters()) + \
                    list(self.dropout.parameters())
        if isinstance(self.lnl_lambda, nn.Parameter):
            target_params.append(self.lnl_lambda)
        domain_params = list(self.domain_classifier.parameters())

                    

        target_optimizer = torch.optim.AdamW(target_params, lr=self.lr, weight_decay=1e-4)
        domain_optimizer = torch.optim.AdamW(domain_params, lr=self.lr, weight_decay=1e-4)
        target_optimizer_scheduler = None
        domain_optimizer_scheduler = None

        if self.scheduler_name=='CosineAnnealingWarmUpRestarts':
            target_optimizer_scheduler = CosineAnnealingWarmUpRestarts(target_optimizer, T_0=50, 
                                                                T_mult=1.25, eta_max=0.1,  
                                                                T_up=10, gamma=0.75, last_epoch=-1)
            domain_optimizer_scheduler = CosineAnnealingWarmUpRestarts(domain_optimizer, T_0=50, 
                                                                T_mult=1.25, eta_max=0.1,  
                                                                T_up=10, gamma=0.75, last_epoch=-1)
        elif self.scheduler_name=='ReduceLROnPlateau':
            target_optimizer_scheduler = ReduceLROnPlateau(target_optimizer,mode='max',factor=0.1, patience=10, threshold=0.0001,cooldown=10)
            domain_optimizer_scheduler = ReduceLROnPlateau(domain_optimizer,mode='min',factor=0.1, patience=10, threshold=0.0001,cooldown=10)
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

    def weight_init(self, m):
        from src.models.components.Conv2d import DepthwiseConv2d, PointwiseConv2d
        if isinstance(m, DepthwiseConv2d) or isinstance(m, PointwiseConv2d):
            nn.init.xavier_uniform_(m.conv.weight)
            if isinstance(m, DepthwiseConv2d):
                with torch.no_grad():
                    norm = m.conv.weight.data.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=1.0)
                    m.conv.weight.data *= desired / (1e-6 + norm)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        # Linear layers 초기화
        elif isinstance(m, nn.Linear):
            # AutoCorrelation의 projection layer들에 대한 특별한 초기화
            if hasattr(self, 'auto_correlation') and self._is_autocorr_projection(m):
                # Query, Key, Value projection: Xavier 초기화
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                # 일반 Linear layer: Xavier 초기화 + norm constraint
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                with torch.no_grad():
                    norm = m.weight.data.norm(2, dim=1, keepdim=True)
                    desired = torch.clamp(norm, max=self.norm_rate)
                    m.weight.data *= desired / (1e-6 + norm)
        
        # AutoCorrelation 특수 파라미터 초기화
        elif hasattr(m, '__class__') and 'AutoCorrelation' in m.__class__.__name__:
            self._init_autocorrelation_params(m)
        
        # Dropout layers (no parameters to initialize)
        elif isinstance(m, nn.Dropout):
            pass
        
        # BatchNorm, LayerNorm 등
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _is_autocorr_projection(self, module):
        """AutoCorrelation의 projection layer인지 확인"""
        if not hasattr(self, 'auto_correlation'):
            return False
        
        # AutoCorrelationLayer의 projection들 확인
        autocorr_layer = self.auto_correlation
        projection_modules = [
            autocorr_layer.query_projection,
            autocorr_layer.key_projection, 
            autocorr_layer.value_projection,
            autocorr_layer.out_projection
        ]
        
        return module in projection_modules

    def _init_autocorrelation_params(self, autocorr_module):
        """AutoCorrelation 모듈의 특수 파라미터 초기화"""
        
        # Pattern weights 초기화 (learnable pattern weights)
        if hasattr(autocorr_module, 'pattern_weights'):
            # 1.0으로 초기화하여 identity mapping부터 시작
            nn.init.ones_(autocorr_module.pattern_weights)
            # 또는 약간의 노이즈를 추가한 초기화
            # nn.init.normal_(autocorr_module.pattern_weights, mean=1.0, std=0.1)
        
        # 기타 learnable parameter들 초기화
        if hasattr(autocorr_module, 'scale') and isinstance(autocorr_module.scale, nn.Parameter):
            nn.init.ones_(autocorr_module.scale)
        
        # Dropout 파라미터는 초기화할 필요 없음
        print(f"AutoCorrelation 파라미터 초기화 완료: {autocorr_module.__class__.__name__}")