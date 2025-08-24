# Enhanced imports with project setup utilities
from typing import Optional, List, Union, Dict, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics
from torchmetrics import Accuracy, F1Score

import pytorch_lightning as pl

# Project imports with improved path management
try:
    from utils.project_setup import project_paths
    from src.utils.metrics.confusion_matrix import calc_confusion_matrix
    from src.utils.metrics.macro_average import calc_metrics
    from src.utils.schedulers.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
    from src.models.components.Conv2d import DepthwiseConv2d, PointwiseConv2d
    from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
    from src.models.components.normalization import ChannelNormLayer
except ImportError as e:
    # Fallback to legacy imports
    import os
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    
    from src.utils.metrics.confusion_matrix import calc_confusion_matrix
    from src.utils.metrics.macro_average import calc_metrics
    from src.utils.schedulers.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
    from src.models.components.Conv2d import DepthwiseConv2d, PointwiseConv2d
    from src.models.components.EEGFeatureExtractor import EEGFeatureExtractor
    from src.models.components.normalization import ChannelNormLayer

# Setup logging
logger = logging.getLogger(__name__)

class BaseEEGNet(pl.LightningModule):
    """Enhanced Base EEGNet with improved error handling and logging.
    
    Args:
        nb_classes: Number of output classes
        norm_rate: Normalization rate for max-norm constraint
        lr: Learning rate
        scheduler_name: Name of learning rate scheduler
        target_monitor: Metric to monitor for scheduling
        channel_norm: Whether to apply channel normalization
        class_weight: Class weights for imbalanced datasets
    """
    
    def __init__(
        self, 
        nb_classes: int = 2, 
        norm_rate: float = 0.25, 
        lr: float = 1e-3,
        scheduler_name: str = 'ReduceLROnPlateau', 
        target_monitor: str = 'val_macro_acc',
        channel_norm: bool = False, 
        class_weight: Optional[List[float]] = None
    ):
        super(BaseEEGNet, self).__init__()
        
        # Save hyperparameters for better reproducibility
        self.save_hyperparameters()

        # Store configuration
        self.lr = lr
        self.norm_rate = norm_rate
        self.scheduler_name = scheduler_name
        self.target_monitor = target_monitor
        self.nb_classes = nb_classes
        
        # Initialize components
        self.channel_norm_layer = ChannelNormLayer(norm_rate=norm_rate) if channel_norm else None
        
        logger.info(f"Initializing BaseEEGNet - Classes: {nb_classes}, LR: {lr}, Scheduler: {scheduler_name}")
        
        # Initialize comprehensive metrics collection
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "macro_acc": torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=nb_classes, average='macro'
                ),
                "macro_f1": torchmetrics.classification.F1Score(
                    task="multiclass", num_classes=nb_classes, average='macro'
                ),
                "micro_acc": torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=nb_classes, average='micro'
                ),
                "micro_f1": torchmetrics.classification.F1Score(
                    task="multiclass", num_classes=nb_classes, average='micro'
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.y_true = []
        self.y_pred = []

        if class_weight is not None:
            weights = torch.tensor(class_weight, dtype=torch.float)
        else:
            weights = None
        self.criterion = nn.CrossEntropyLoss(weight=weights)
    
    def _update_y_info(self, labels, preds):
        self.y_true.append(labels)
        self.y_pred.append(preds)
    
    def _reset_y_info(self):
        self.y_true.clear()
        self.y_pred.clear()

    def _save_report(self, labels, preds, prefix='train'):
        flat_report = calc_metrics(labels, preds, 
                                   prefix=prefix, target_names=['stim_before', 'stim_after'])
        # Log the flattened report
        for key, value in flat_report.items():
            self.log(key, value, on_epoch=True, sync_dist=True)
    
    def _save_confusion_matrix(self, prefix='test'):
        print(f">> self.y_true: {self.y_true}")
        print(f">> self.y_pred: {self.y_pred}")
        conf_matrix = calc_confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        n_class = conf_matrix.shape[0]
        for i in range(n_class):
            for j in range(n_class):
                self.log(f"{prefix}_CM/{i}_{j}", float(conf_matrix[i, j]), on_epoch=True, sync_dist=True)
    
    def _step_func(self, batch, prefix='train'):
        x, labels = batch

        if isinstance(labels, tuple) or isinstance(labels, list):
            y, _ = labels
            labels = y
    
        output = self(x)

        loss = self.criterion(output, labels)
        preds = torch.argmax(output, dim=1)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True)

        getattr(self, f'{prefix}_metrics').update(preds, labels)

        return loss, labels, preds
        # self._update_y_info(labels, preds)
    
    def _epoch_end_func(self, prefix='train'):
        tmp_metrics = getattr(self, f'{prefix}_metrics')
        metrics_results = tmp_metrics.compute()
        tmp_metrics.reset()

        self.log_dict(metrics_results, on_epoch=True ,prog_bar=True, sync_dist=True)

    
    def training_step(self, batch, batch_idx):
        loss, labels, preds = self._step_func(batch, 'train')
        self._save_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'train')
        return loss


    def validation_step(self, batch, batch_idx):
        loss, labels, preds = self._step_func(batch, 'val')
        self._save_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'val')

    def test_step(self, batch, batch_idx):
        loss, labels, preds = self._step_func(batch, 'test')
        self._update_y_info(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.apply_max_norm()
    
    def on_train_epoch_end(self):
        self._epoch_end_func('train')

    def on_validation_epoch_end(self):
        self._epoch_end_func('val')

    def on_test_epoch_end(self):
        self._epoch_end_func('test')
        self._save_confusion_matrix()
        self._save_report(self.y_true, self.y_pred, 'test')

        self._reset_y_info()
    
    def weight_init(self, m):
        if isinstance(m, DepthwiseConv2d) or isinstance(m, PointwiseConv2d):
            nn.init.xavier_uniform_(m.conv.weight)
            if isinstance(m, DepthwiseConv2d):
                with torch.no_grad():
                    norm = m.conv.weight.data.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=1.0)
                    m.conv.weight.data *= desired / (1e-6 + norm)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                norm = m.weight.data.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=self.norm_rate)
                m.weight.data *= desired / (1e-6 + norm)

    def apply_max_norm(self):
        # classifier가 반드시 두 번째 레이어(nn.Linear)를 포함해야 함
        with torch.no_grad():
            weight = self.classifier[1].weight
            norm = weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            weight.data *= desired / (1e-6 + norm)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer_scheduler = None

        if self.scheduler_name=='CosineAnnealingWarmUpRestarts':
            optimizer_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, 
                                                                T_mult=1.25, eta_max=0.1,  
                                                                T_up=10, gamma=0.75, last_epoch=-1)
        elif self.scheduler_name=='ReduceLROnPlateau':
            optimizer_scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5, patience=5, threshold=0.001,cooldown=10)
        else:
            print(">> Not use LR scheduler?")

        if optimizer_scheduler:
            return [ 
                {'optimizer': optimizer, 
                 'lr_scheduler': {'scheduler':optimizer_scheduler, 'name':'target_lr', 'monitor':self.target_monitor}},
            ]
        else:
            return optimizer


from src.utils.domain_adaptation import GradientReversal

class BaseEEGNetGRL(pl.LightningModule):
    def __init__(self, nb_classes=2, domain_classes=3, norm_rate=0.25, lr=1e-3, grl_lambda=0.3,
                 scheduler_name='ReduceLROnPlateau', target_monitor='val_target_macro_acc', domain_monitor='val_domain_macro_acc',
                 channel_norm=False, class_weight=None, domain_loss_type='CE', soft_label=False):
        super(BaseEEGNetGRL, self).__init__()
        self.automatic_optimization=False

        self.lr = lr
        self.norm_rate = norm_rate

        self.soft_label = soft_label

        self.scheduler_name = scheduler_name
        self.target_monitor = target_monitor
        self.domain_monitor = domain_monitor

        self.channel_norm_layer = ChannelNormLayer(norm_rate=norm_rate) if channel_norm else None
        
        # 공통 메트릭
        self.train_target_metrics = torchmetrics.MetricCollection(
            {
            "macro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=nb_classes, average='macro'
            ),
            "macro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=nb_classes, average='macro'
            ),
            "micro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=nb_classes, average='micro'
            ),
            "micro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=nb_classes, average='micro'
            ),
            },
            prefix="train_target_",  # 메트릭 이름에 접두어
        )
        self.train_domain_metrics = torchmetrics.MetricCollection(
            {
            "macro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=domain_classes, average='macro'
            ),
            "macro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=domain_classes, average='macro'
            ),
            "micro_acc": torchmetrics.classification.Accuracy(
                task="multiclass", num_classes=domain_classes, average='micro'
            ),
            "micro_f1": torchmetrics.classification.F1Score(
                task="multiclass", num_classes=domain_classes, average='micro'
            ),
            },
            prefix="train_target_",  # 메트릭 이름에 접두어
        )

        self.val_target_metrics = self.train_target_metrics.clone(prefix="val_target_")
        self.val_domain_metrics = self.train_domain_metrics.clone(prefix="val_domain_")
        
        self.test_target_metrics = self.train_target_metrics.clone(prefix="test_target_")

        # GRL layer
        self.grl = GradientReversal(lambda_=grl_lambda)

        if class_weight is not None:
            weights = torch.tensor(class_weight, dtype=torch.float)
        else:
            weights = None
        self.target_criterion = nn.CrossEntropyLoss(weight=weights)

        self.domain_loss_type = domain_loss_type
        if soft_label:
            if self.domain_loss_type == 'KLDiv':
                self.domain_criterion = nn.KLDivLoss(reduction='batchmean')
            else:
                self.domain_criterion = nn.CrossEntropyLoss()
            self.ce_domain_criterion = nn.CrossEntropyLoss()
        else:
            self.domain_criterion = nn.CrossEntropyLoss()

        self.y_true = []
        self.y_pred = []
    
    def _update_y_info(self, labels, preds):
        self.y_true.append(labels)
        self.y_pred.append(preds)
    
    def _reset_y_info(self):
        self.y_true.clear()
        self.y_pred.clear()

    def _save_report(self, labels, preds, prefix='train'):
        flat_report = calc_metrics(labels, preds, 
                                   prefix=prefix, target_names=['stim_before', 'stim_after'])
        # Log the flattened report
        for key, value in flat_report.items():
            self.log(key, value, on_epoch=True, sync_dist=True)
    
    def _save_confusion_matrix(self, prefix='test'):
        conf_matrix = calc_confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        n_class = conf_matrix.shape[0]
        for i in range(n_class):
            for j in range(n_class):
                self.log(f"{prefix}_CM/{i}_{j}", float(conf_matrix[i, j]), on_epoch=True, sync_dist=True)
    
    def _calc_loss_pseudo_pred_domain(
        self,
        probs: torch.Tensor,   # raw logits (B, domain_classes)
        target_labels: torch.Tensor    # (B,) class indices
    ) -> torch.Tensor:
        """
        Mutual-information–형 pseudo-domain loss를 계산한다.
        • self.lnl_lambda 가 nn.Parameter(클래스별 learnable λ) 인 경우 → 샘플마다 가중치 적용
        • 그렇지 않은 경우(스칼라 고정 값) → MI 평균에 λ 상수를 곱해 반환
        """

        # sample-wise MI   (B,)
        eps = 1e-8
        mi_per_sample = torch.sum(probs * torch.log(probs + eps), dim=1)

        # ───────── λ 적용 방식 분기 ─────────
        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            # 클래스별 learnable λ            shape = [nb_classes]
            lam_batch = self.lnl_lambda[target_labels]      # (B,)
            loss = (lam_batch * mi_per_sample).mean()
        else:
            # 고정 스칼라 λ
            loss = mi_per_sample.mean() * self.lnl_lambda

        return loss
    
    def _calc_first_step_loss(self, target_loss, loss_pseudo_pred_domain):
        return target_loss

    def _first_step(self, x, target_labels, target_optimizer, domain_optimizer, prefix='train'):
        # first step
        target_output, domain_output = self(x)
        target_loss = self.target_criterion(target_output, target_labels)
        domain_output = self.domain_softmax(domain_output)

        epsilon = 1e-8
        # loss_pseudo_pred_domain = torch.mean(torch.sum(domain_output*torch.log(domain_output+epsilon),1))
        loss_pseudo_pred_domain = self._calc_loss_pseudo_pred_domain(domain_output, target_labels)
        # first_step_total_loss = target_loss + loss_pseudo_pred_domain*self.lnl_lambda
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
        target_output, domain_output = self(x, second_step=True)
        if self.soft_label:
            if prefix=='train':
                if self.domain_loss_type == 'KLDiv':
                    domain_loss = self.domain_criterion(F.log_softmax(domain_output, dim=1), domain_labels)
                else:
                    domain_loss = self.domain_criterion(domain_output, domain_labels)
            else:
                domain_loss = self.ce_domain_criterion(domain_output, domain_labels)
        else:
            domain_loss = self.domain_criterion(domain_output, domain_labels)

        # optim only domain classifier
        if prefix=='train':
            target_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            self.manual_backward(domain_loss)
            target_optimizer.step()
            domain_optimizer.step()
                
        # 예측 및 정확도 계산
        domain_preds = torch.argmax(domain_output, dim=1)

        if self.soft_label:
            if prefix!='train':
                getattr(self, f'{prefix}_domain_metrics').update(domain_preds, domain_labels)
        else:
            getattr(self, f'{prefix}_domain_metrics').update(domain_preds, domain_labels)
                    
                

        self.log(f'{prefix}_domain_loss', domain_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return domain_labels, domain_preds

    def _step_func(self, batch, prefix='train'):
        target_optimizer, domain_optimizer = self.optimizers()

        x, (target_labels, domain_labels) = batch  # 배치는 (데이터, 타겟 레이블, 도메인 레이블) 형태

        # first step
        target_labels, target_preds = self._first_step(x, target_labels, target_optimizer, domain_optimizer, prefix)

        # second step
        domain_labels, domain_preds = self._second_step(x, domain_labels, target_optimizer, domain_optimizer, prefix)

        return target_labels, target_preds

    def _epoch_end_func(self, prefix='train'):
        tmp_target_metrics = getattr(self, f'{prefix}_target_metrics')
        tmp_domain_metrics = getattr(self, f'{prefix}_domain_metrics')

        target_metrics_results = tmp_target_metrics.compute()
        domain_metrics_results = tmp_domain_metrics.compute()
        tmp_target_metrics.reset()
        tmp_domain_metrics.reset()

        self.log_dict(target_metrics_results, on_epoch=True ,prog_bar=True, sync_dist=True)
        self.log_dict(domain_metrics_results, on_epoch=True ,prog_bar=True, sync_dist=True)

        if isinstance(self.lnl_lambda, torch.nn.Parameter):
            # 클래스별 learnable λ            
            for i in range(len(self.lnl_lambda)):
                self.log(f"{prefix}_lnl_learnable_lambda_cls{i}", self.lnl_lambda[i], prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            # 고정 스칼라 λ
            self.log(f'{prefix}_lnl_lambda', self.lnl_lambda, on_epoch=True, prog_bar=True, sync_dist=True)

        # self._save_report(self.y_true, self.y_pred, prefix)
    
    def _lr_scheduler_func(self):
        target_optimizer_scheduler, domain_optimizer_scheduler = self.lr_schedulers()
        if self.scheduler_name == 'ReduceLROnPlateau':
            target_optimizer_scheduler.step(self.trainer.callback_metrics[self.target_monitor])
            domain_optimizer_scheduler.step(self.trainer.callback_metrics[self.domain_monitor])
        else:
            target_optimizer_scheduler.step()
            domain_optimizer_scheduler.step()    


    def training_step(self, batch, batch_idx):
        labels, preds = self._step_func(batch, 'train')
        self._save_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'train')


    def validation_step(self, batch, batch_idx):
        labels, preds = self._step_func(batch, 'val')
        self._save_report(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), 'val')

    def test_step(self, batch, batch_idx):
        x, target_labels = batch
        target_output, _ = self(x)

        target_loss = self.target_criterion(target_output, target_labels)
        target_preds = torch.argmax(target_output, dim=1)

        self.log('test_target_loss', target_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        getattr(self, 'test_target_metrics').update(target_preds, target_labels)
        self._update_y_info(target_labels.detach().cpu().numpy(), target_preds.detach().cpu().numpy())
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.apply_max_norm()

    def on_train_epoch_end(self):
        self._epoch_end_func('train')

    def on_validation_epoch_end(self):
        self._epoch_end_func('val')        

    def on_test_epoch_end(self):
        target_metrics_results = self.test_target_metrics.compute()
        self.test_target_metrics.reset()
        self.log_dict(target_metrics_results, on_epoch=True ,prog_bar=True, sync_dist=True)

        self._save_confusion_matrix()
        self._save_report(self.y_true, self.y_pred, 'test')
        self._reset_y_info()

    def on_validation_end(self):
        self._lr_scheduler_func()

    def weight_init(self, m):
        if isinstance(m, DepthwiseConv2d) or isinstance(m, PointwiseConv2d):
            nn.init.xavier_uniform_(m.conv.weight)
            if isinstance(m, DepthwiseConv2d):
                with torch.no_grad():
                    norm = m.conv.weight.data.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=1.0)
                    m.conv.weight.data *= desired / (1e-6 + norm)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                norm = m.weight.data.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=self.norm_rate)
                m.weight.data *= desired / (1e-6 + norm)

    def apply_max_norm(self):
        # classifier가 반드시 두 번째 레이어(nn.Linear)를 포함해야 함
        with torch.no_grad():
            weight = self.target_classifier[1].weight
            norm = weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            weight.data *= desired / (1e-6 + norm)
    
    def configure_optimizers(self):
        target_optimizer = torch.optim.Adam(list(self.feature_extractor.parameters())+list(self.target_classifier.parameters()), lr=self.lr)
        domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=self.lr)
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
    


