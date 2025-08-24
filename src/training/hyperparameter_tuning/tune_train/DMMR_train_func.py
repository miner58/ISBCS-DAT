"""
DMMR Training Function for Ray Tune Integration

Implements 2-stage DMMR training (pretraining → finetuning) with seamless Ray Tune integration.
Follows the existing base_train_func.py patterns for compatibility.
"""

import os
import sys
import random
import json
import copy
from typing import Dict, Any, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything

# Ray + Lightning 관련
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

try :
    # 사용자 정의 모듈
    print("Importing DMMR modules... in DMMR_train_func.py")
    from src.data.modules.dmmr_compatible_datamodule import DMMRCompatibleDataModule, create_dmmr_datamodule
    print("DMMR data modules imported successfully.")
    from src.models.dmmr_adapter import DMMRPreTraining, DMMRFineTuning, create_dmmr_model
    print("DMMR adapter models imported successfully.")
except ImportError as e:
    raise ImportError(f"Failed to import DMMR modules: {e}")

class DMMRLightningRunner:
    """
    2-stage DMMR training runner for Ray Tune integration.
    
    Handles:
    1. Pre-training phase: Self-supervised reconstruction + domain adversarial learning
    2. Fine-tuning phase: Classification with pre-trained features
    3. Seamless Ray Tune integration with metrics reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DMMR training runner.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.data_module = None
        self.pretraining_model = None
        self.finetuning_model = None
        self.trainer = None
        
        # DMMR-specific config
        self.dmmr_config = self._setup_dmmr_config()
        
        # Set seed if specified
        if config.get("fix_seed", False):
            self._set_seed(config["seed"])
    
    def _setup_dmmr_config(self) -> Dict[str, Any]:
        """Setup DMMR-specific configuration from general config."""
        dmmr_config = {
            # Extract target subject from subject_name
            'target_subject': self.config.get('subject_name', 'B202'),
            
            # DMMR hyperparameters (fixed epochs like original paper)
            'beta': self.config.get('dmmr_beta', 1.0),
            'pretraining_epochs': self.config.get('dmmr_pretraining_epochs', 100),  # Original DMMR default
            'finetuning_epochs': self.config.get('dmmr_finetuning_epochs', 50),   # Original DMMR default
            'pretraining_lr': self.config.get('dmmr_pretraining_lr', self.config.get('lr', 1e-3)),
            'finetuning_lr': self.config.get('dmmr_finetuning_lr', self.config.get('lr', 1e-4)),
            
            # Model parameters
            'hidden_dim': self.config.get('dmmr_hidden_dim', 64),
            # 참고: 원본 DMMR 논문에서는 freeze_pretrained=False를 사용
            # DMMRFineTuningModule에서 자동으로 False로 강제 설정됨
            'freeze_pretrained': self.config.get('dmmr_freeze_pretrained', False),
            
            # 🆕 Additional DMMR parameters from config
            'window_size': self.config.get('dmmr_window_size', 6),
            'time_steps': self.config.get('dmmr_time_steps', 6),
            'input_dim': self.config.get('dmmr_input_dim', 310),
            'n_layers': self.config.get('dmmr_n_layers', 1),
            'weight_decay': self.config.get('dmmr_weight_decay', 0.0005),
        }
        
        # DMMR configuration loaded
        
        return dmmr_config
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_everything(seed, workers=True)
    
    def _build_data_module(self):
        """Build DMMR-compatible data module."""
        
        # Handle skip_time_list if specified
        skip_time_list = None
        if self.config.get("skip_time_list"):
            with open(self.config["skip_time_list"] + ".json", 'r') as f:
                skip_time_list = json.load(f)
        
        # Build data config path
        data_config_path = self.config["data_config"] + self.config.get('subject_name', "") + ".json"
        
        # 🆕 Check for cross-domain experiment
        is_cross_domain = self.config.get('subject_usage') == 'cross_domain'
        
        # Cross-domain experiment handling
        if is_cross_domain and 'test_data_config' not in self.config:
            print("Warning: test_data_config not specified for cross-domain experiment")
        
        # Load data configuration
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)
        
        print(f"data_config_path exit: {os.path.exists(data_config_path)}")
        # Create DMMR-compatible DataModule
        self.data_module = DMMRCompatibleDataModule(
            data_config=data_config,
            batch_size=self.config['batch_size'],
            masking_ch_list=self.config.get('masking_ch_list', []),
            rm_ch_list=self.config.get('rm_ch_list', []),
            subject_usage=self.config.get('subject_usage', 'all'),
            seed=self.config['seed'] if self.config.get('fix_seed', False) else None,
            default_path=self.config['data_default_path'],
            skip_time_list=skip_time_list,
            # DMMR-specific parameters
            dmmr_mode=True,
            time_steps=self.dmmr_config['time_steps'],
        )
        
    def _build_pretraining_model(self):
        """Build DMMR pre-training model."""
        dmmr_params = self.data_module.dmmr_params
        print(f"🔍 DMMR parameters: {dmmr_params}")
        
        # Build model configuration with only necessary parameters
        model_config = {
            'nb_classes': self.config['nb_classes'],
            'lr': self.dmmr_config['pretraining_lr'],
            'class_weight': self.config.get('class_weight'),
            'scheduler_type': self.config.get('scheduler'),
            'optimizer_type': 'adam',
            'beta': self.dmmr_config['beta'],
            'hidden_dim': self.dmmr_config['hidden_dim'],
            'dropoutRate': self.config.get('dropoutRate', 0.0),
            # DMMR model parameters from data module
            'number_of_source': dmmr_params.get('number_of_source', 2),
            'batch_size': dmmr_params.get('batch_size', 10),
            'time_steps': dmmr_params.get('time_steps', 15),
            'input_dim': dmmr_params.get('input_dim', 310),
        }
        
        self.pretraining_model = DMMRPreTraining(**model_config)
    
    def _build_finetuning_model(self):
        """Build DMMR fine-tuning model.

        Note: Weight transfer from pre-training model is handled automatically
        by DMMRFineTuning class through pretrained_module parameter.
        No manual weight transfer is required.
        """
        dmmr_params = self.data_module.dmmr_params
        
        # Build model configuration with only necessary parameters
        model_config = {
            'nb_classes': self.config['nb_classes'],
            'lr': self.dmmr_config['finetuning_lr'],
            'class_weight': self.config.get('class_weight'),
            'scheduler_type': self.config.get('scheduler'),
            'optimizer_type': 'adam',
            'hidden_dim': self.dmmr_config['hidden_dim'],
            'dropoutRate': self.config.get('dropoutRate', 0.0),
            # DMMR model parameters from data module
            'number_of_source': dmmr_params.get('number_of_source', 2),
            'batch_size': dmmr_params.get('batch_size', 10),
            'time_steps': dmmr_params.get('time_steps', 15),
            'input_dim': dmmr_params.get('input_dim', 310),
            # Fine-tuning specific parameters
            'pretrained_module': self.pretraining_model,
            'freeze_pretrained': self.dmmr_config.get('freeze_pretrained', True),  # 🔧 설정에서 freeze 여부 확인
        }
        
        self.finetuning_model = DMMRFineTuning(**model_config)

        # 디버깅: weight transfer 상태 확인
        print("✅ Fine-tuning model created with automatic weight transfer")
        
        # Gradient 상태 확인 (선택적)
        total_params = sum(1 for p in self.finetuning_model.parameters() if p.requires_grad)
        print(f"🎯 Fine-tuning model: {total_params} parameters with gradient enabled")
    
    def _create_trainer(self, stage: str, max_epochs: int) -> pl.Trainer:
        """Create Lightning trainer for specific stage."""
        # Setup callbacks
        callbacks = [RayTrainReportCallback()]
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Early stopping - DISABLED for DMMR paper consistency
        # Original DMMR uses fixed epochs without early stopping
        # if stage == 'pretraining':
        #     monitor_metric = 'train_total_loss'
        #     mode = 'min'
        # else:
        #     monitor_metric = self.config.get('monitor_value_name', 'val_macro_acc')
        #     mode = 'max'
        # 
        # early_stop = EarlyStopping(
        #     monitor=monitor_metric,
        #     patience=20 if stage == 'pretraining' else 10,
        #     mode=mode,
        #     verbose=True,
        #     min_delta=1e-3,
        #     check_finite=True
        # )
        # callbacks.append(early_stop)
        
        # Early stopping disabled for DMMR paper consistency
        
        # Create trainer with RayDDPStrategy (required by Ray Train)
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=callbacks,
            plugins=[RayLightningEnvironment()],
            logger=None,
            # enable_progress_bar=True,
            max_epochs=max_epochs,
            # min_epochs removed - use fixed epochs like original DMMR
            # min_epochs=10 if stage == 'pretraining' else 5
            # enable_checkpointing=False,  # Disable checkpointing for paper consistency
            use_distributed_sampler=False,  # Disable Lightning's sampler injection since we use custom batch sampler,
            deterministic=True
        )
        
        # Prepare for Ray
        return prepare_trainer(trainer)
    
    def _run_pretraining(self):
        """Run DMMR pre-training phase."""
        print("Starting DMMR pre-training phase...")
        
        # setup data module for pre-training
        self.data_module.setup('fit', 'pretraining')

        # Create trainer
        trainer = self._create_trainer('pretraining', self.dmmr_config['pretraining_epochs'])
        
        # Train model
        trainer.fit(self.pretraining_model, self.data_module.train_dataloader(), self.data_module.val_dataloader())
        
        print("Pre-training phase completed.")
        
        return trainer
    
    def _run_finetuning(self):
        """Run DMMR fine-tuning phase."""
        print("Starting DMMR fine-tuning phase...")
        
        # setup data module for fine-tuning
        self.data_module.setup('fit', 'finetuning')

        # Create trainer
        trainer = self._create_trainer('finetuning', self.dmmr_config['finetuning_epochs'])
        
        # Train model
        trainer.fit(self.finetuning_model, self.data_module.train_dataloader(), self.data_module.val_dataloader())
        
        print("Fine-tuning phase completed.")
        
        return trainer
    
    def train(self):
        """
        Execute complete DMMR training pipeline.
        
        Returns:
            Final training metrics
        """
        print("Starting DMMR 2-stage training pipeline...")
        
        try:
            # Step 1: Build data module
            self._build_data_module()
            
            # Step 2: Build models
            self._build_pretraining_model()
            self._build_finetuning_model()
            
            # Step 3: Run pre-training
            pretraining_trainer = self._run_pretraining()
            
            # Step 4: Run fine-tuning
            finetuning_trainer = self._run_finetuning()
            
            print("DMMR training pipeline completed successfully.")
            
            # Return final metrics (Ray Tune will handle reporting via callbacks)
            return {"status": "completed", "stage": "finetuning"}
        
        except Exception as e:
            print(f"DMMR Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise e


def DMMR_train_func(config: Dict[str, Any]):
    """
    DMMR training function for Ray Tune integration.
    
    This function follows the same pattern as base_train_func.py but implements
    the 2-stage DMMR training process.
    
    Args:
        config: Training configuration from Ray Tune
    """
    print("DMMR training function started.")
    
    # Create and run DMMR training pipeline
    runner = DMMRLightningRunner(config)
    result = runner.train()
    
    print(f"DMMR training completed: {result}")
    return result


def test_func(config: Dict[str, Any], checkpoint: str, sub_config_path: str, test_data_default_path: str):
    """
    DMMR test function for Ray Tune integration with cross-domain support.
    
    Args:
        config: Test configuration
        checkpoint: Path to trained model checkpoint
        sub_config_path: Path to test data configuration
        test_data_default_path: Default path for test data
        
    Returns:
        Test results
    """
    print("DMMR test function started.")
    
    try:
        # Load test configuration
        with open(sub_config_path, 'r') as f:
            test_data_config = json.load(f)
        
        # Create test data module with cross-domain support
        datamodule_kwargs = {
            'data_config': test_data_config,
            'batch_size': config['batch_size'],
            'masking_ch_list': config.get('masking_ch_list', []),
            'rm_ch_list': config.get('rm_ch_list', []),
            'subject_usage': config.get('subject_usage', 'all'),
            'seed': config['seed'] if config.get('fix_seed', False) else None,
            'default_path': test_data_default_path,
            'dmmr_mode': True,
            'time_steps': config.get('dmmr_time_steps', 6),
        }
        
        test_datamodule = DMMRCompatibleDataModule(**datamodule_kwargs)
        test_datamodule.setup('test')
        
        # Load trained model (use fine-tuning model for testing)
        from src.models.dmmr import DMMRFineTuningModule
        model = DMMRFineTuningModule.load_from_checkpoint(
            os.path.join(checkpoint, "checkpoint.ckpt")
        )
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # Create test trainer with minimal configuration
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            logger=None,
            enable_progress_bar=False,
            use_distributed_sampler=False,  # Disable Lightning's sampler injection since we use custom batch sampler
        )
        
        # Run test using the explicit test_step method
        test_results = trainer.test(model, test_datamodule.test_dataloader(), verbose=False)

        return test_results  # Return results for compatibility with existing analysis code
        
        # # Extract key metrics from test results
        # if test_results and len(test_results) > 0:
        #     result = test_results[0]
            
        #     # Create standardized result format for compatibility with existing analysis code
        #     standardized_result = {
        #         'test_acc': result.get('test_acc', result.get('test/accuracy', 0.0)),
        #         'test_loss': result.get('test_loss', result.get('test/loss', 0.0)),
        #         'test_macro_acc': result.get('test_macro_acc', result.get('test/macro_accuracy', result.get('test_acc', 0.0))),
        #         'test_micro_acc': result.get('test_micro_acc', result.get('test/micro_accuracy', result.get('test_acc', 0.0))),
        #         'test_f1': result.get('test_f1', result.get('test/f1_score', 0.0)),
        #         'subjects_processed': result.get('test_subjects_processed', 1),
        #         'batch_size': result.get('test_batch_size', config['batch_size']),
        #     }
            
        #     # Add report-style metrics for compatibility
        #     standardized_result['test/report/macro avg/accuracy'] = standardized_result['test_macro_acc']
        #     standardized_result['test/report/micro avg/accuracy'] = standardized_result['test_micro_acc']
            
        # else:
        #     # Fallback result if test fails
        #     standardized_result = {
        #         'test_acc': 0.0,
        #         'test_loss': float('inf'),
        #         'test_macro_acc': 0.0,
        #         'test_micro_acc': 0.0,
        #         'test_f1': 0.0,
        #         'test/report/macro avg/accuracy': 0.0,
        #         'test/report/micro avg/accuracy': 0.0,
        #         'error': 'Test execution failed'
        #     }
        
        # print(f"🧪 DMMR test completed: {standardized_result}")
        # return [standardized_result]  # Return as list for compatibility with existing analysis code
    
    except Exception as e:
        print(f"DMMR test failed: {e}")
        import traceback
        traceback.print_exc()
        return [{"error": str(e), "test_acc": 0.0, "test_loss": float('inf')}]


# Utility functions for DMMR configuration
def create_dmmr_config_template(base_config: Dict[str, Any], dataset: str = 'raw3') -> Dict[str, Any]:
    """
    Create DMMR configuration template based on dataset.
    
    Args:
        base_config: Base configuration dictionary
        dataset: Dataset name ('raw3', 'raw5and6', 'UI', 'UNM')
        
    Returns:
        DMMR-optimized configuration
    """
    dmmr_config = copy.deepcopy(base_config)
    
    # Dataset-specific optimizations
    if dataset in ['raw3', 'raw5and6']:
        # Small mouse datasets
        dmmr_config.update({
            'dmmr_beta': 1.0,
            'dmmr_pretraining_epochs': 80,
            'dmmr_finetuning_epochs': 40,
            'dmmr_hidden_dim': 64,
            'batch_size': 10
        })
    elif dataset in ['UI', 'UNM']:
        # Larger human datasets
        dmmr_config.update({
            'dmmr_beta': 5.0,
            'dmmr_pretraining_epochs': 120,
            'dmmr_finetuning_epochs': 60,
            'dmmr_hidden_dim': 128,
            'batch_size': 16
        })
    
    return dmmr_config