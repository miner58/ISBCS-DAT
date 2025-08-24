"""
Feature Extractor Module

체크포인트에서 특징을 추출하고 저장하는 모듈
"""

import os
import json
import yaml
import glob
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from src.models.eegnet import EEGNet
from src.models.eegnet_grl import EEGNetGRL, EEGNetLNL, EEGNetMI
from src.models.eegnetDRO import EEGNetDRO
from src.models.eegnet_grl_lag import EEGNetLNLLag
from src.models.dmmr import DMMRFineTuningModule, DMMRPreTrainingModule
from src.data.modules.EEGdataModuel import EEGDataModule


@dataclass
class FeatureData:
    """특징 데이터 컨테이너"""
    features: np.ndarray
    target_labels: np.ndarray
    input_data: np.ndarray
    domain_labels: Optional[np.ndarray] = None
    subject_name: str = ""
    model_name: str = ""
    checkpoint_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionConfig:
    """특징 추출 설정"""
    test_data_path: str
    model_types: List[str] = field(default_factory=lambda: ['EEGNet'])
    batch_size: int = 16
    use_augmentation: bool = False
    augmentation_config_path: Optional[str] = None
    max_extractions: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelRegistry:
    """모델 타입 레지스트리"""
    
    _models = {
        'EEGNet': EEGNet,
        'EEGNetGRL': EEGNetGRL,
        'EEGNetLNL': EEGNetLNL,
        'EEGNetMI': EEGNetMI,
        'EEGNetDRO': EEGNetDRO,
        'EEGNetLNLLag': EEGNetLNLLag,
        'DMMRFineTuningModule': DMMRFineTuningModule,
        'DMMRPreTrainingModule': DMMRPreTrainingModule,
        # Legacy names for compatibility
        'EEGNetDomainAdaptation_LNL': EEGNetLNL,
        'EEGNetDomainAdaptation_Not_GRL': EEGNetMI,
        'EEGNetDomainAdaptation_Only_GRL': EEGNetGRL,
    }
    
    @classmethod
    def get_model_class(cls, model_name: str):
        """모델 클래스 반환"""
        if model_name not in cls._models:
            raise ValueError(f"지원하지 않는 모델: {model_name}. 지원 모델: {list(cls._models.keys())}")
        return cls._models[model_name]
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """새 모델 등록"""
        cls._models[name] = model_class


from src.data.modules.EEGdataModuel import AugmentedCollateFunction
from torch.utils.data import DataLoader
class EEGDataModuleAugTest(EEGDataModule):
    def __init__(self, data_config: dict, batch_size: int = 16, masking_ch_list=None, rm_ch_list=None, 
                subject_usage: str = "all", seed: int = 42, skip_time_list: dict = None, 
                default_path: str = "/home/jsw/Fairness/Fairness_for_generalization",
                data_augmentation_config: dict = None):
        super().__init__(data_config, batch_size, masking_ch_list, rm_ch_list, subject_usage, seed, skip_time_list, default_path, data_augmentation_config=data_augmentation_config)

    def test_dataloader(self):
        # Apply augmentation only to training data by default
        collate_fn = None
        train_augmentation = None
        apply_augmentation = self.data_augmentation_config.get('enabled', False)
        if apply_augmentation:
            train_augmentation = self.data_augmentation_pipeline if self.data_augmentation_config.get('train_only', True) else self.data_augmentation_pipeline
            # Create custom collate function with augmentation
            collate_fn = AugmentedCollateFunction(
                augmentation=train_augmentation,
                apply_augmentation= apply_augmentation
            )
        
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False,
            collate_fn=collate_fn
        )

class DataModuleFactory:
    """데이터 모듈 생성 팩토리"""
    
    @staticmethod
    def create_data_module(
        test_config_path: str,
        batch_size: int,
        default_path: str,
        data_augmentation_config: Optional[Dict] = None
    ) -> EEGDataModuleAugTest:
        """데이터 모듈 생성"""
        # 경로 정규화
        if not os.path.isabs(test_config_path):
            test_config_path = os.path.join(default_path, test_config_path)
        
        if not os.path.exists(test_config_path):
            raise FileNotFoundError(f"데이터 설정 파일이 없습니다: {test_config_path}")
        
        with open(test_config_path, 'r') as f:
            data_config = json.load(f)
        
        data_module = EEGDataModuleAugTest(
            data_config=data_config,
            batch_size=batch_size,
            masking_ch_list=[],
            rm_ch_list=[],
            subject_usage="test1",
            seed=None,
            default_path=default_path,
            skip_time_list=None,
            data_augmentation_config=data_augmentation_config
        )
        
        data_module.setup('test')
        return data_module


class FeatureExtractor:
    """체크포인트 기반 특징 추출기"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_model_from_checkpoint(self, checkpoint_path: str, model_name: str):
        """체크포인트에서 모델 로드"""
        model_class = ModelRegistry.get_model_class(model_name)
        
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.ckpt")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_file}")
        
        model = model_class.load_from_checkpoint(ckpt_file)
        model.eval()
        model.to(self.device)
        return model
    
    def create_augmentation_config(self, aug_config_path: str) -> Dict:
        """데이터 증강 설정 생성"""
        if not aug_config_path:
            return {'enabled': False}
        
        full_path = os.path.join(self.config.test_data_path, aug_config_path)
        
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Ray Tune 스타일 설정 변환
            processed_config = {}
            for key, value in config.items():
                if isinstance(value, dict) and 'value' in value:
                    processed_config[key] = value['value']
                else:
                    processed_config[key] = value
            
            return self._prepare_augmentation_config(processed_config)
        except Exception as e:
            print(f"데이터 증강 설정 로드 실패: {e}")
            return {'enabled': False}
    
    def _prepare_augmentation_config(self, config: dict) -> Dict:
        """증강 설정 전처리"""
        final_config = {
            'enabled': config.get('enabled', True),
            'train_only': config.get('train_only', True),
            'methods': []
        }
        
        for method_name in config.get('methods', []):
            method_setting = config.get('setting', {}).get(method_name)
            if not method_setting:
                continue
                
            method_info = {
                'type': method_setting.get('name'),
                'prob_method': method_setting.get('swap_probability_method', 'uniform')
            }
            
            if method_name == 'CorticalRegionChannelSwap':
                regions_path = method_setting.get('cortical_regions_path')
                if regions_path:
                    full_regions_path = os.path.join(self.config.test_data_path, regions_path)
                    method_info['regions'] = self._load_cortical_regions(full_regions_path)
            
            elif method_name == 'SubjectLevelChannelSwap':
                method_info['enable_soft_labels'] = method_setting.get('enable_soft_labels', False)
            
            final_config['methods'].append(method_info)
        
        return final_config
    
    def _load_cortical_regions(self, regions_path: str) -> List[List[int]]:
        """cortical regions 로드"""
        with open(regions_path, 'r') as f:
            regions = json.load(f)
        return [values for values in regions.values()]
    
    def extract_features(self, model, data_module) -> FeatureData:
        """모델에서 특징 추출"""
        model.eval()
        
        all_features = []
        all_target_labels = []
        all_domain_labels = []
        all_input_data = []
        
        with torch.no_grad():
            print(f"데이터 모듈 테스트 데이터 로딩: {len(data_module.test_dataloader())} 배치")
            for batch in iter(data_module.test_dataloader()):
                x, labels = batch
                # 배치 언패킹
                if len(labels) == 2:
                    target_labels, domain_labels = labels
                else:
                    target_labels = labels
                    domain_labels = None
                
                # 입력 데이터 저장
                all_input_data.append(x.cpu().numpy())
                
                # GPU로 이동
                x = x.to(self.device)

                # 특징 추출
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                else:
                    # Fallback: feature_extractor 사용
                    features = model.feature_extractor(x.permute(0, 3, 1, 2))
                    features = torch.flatten(features, start_dim=1)
                
                all_features.append(features.cpu().numpy())
                all_target_labels.append(target_labels.numpy())
                
                if domain_labels is not None:
                    all_domain_labels.append(domain_labels.numpy())
        
        # 결과 조합
        features_array = np.vstack(all_features)
        target_labels_array = np.hstack(all_target_labels)
        domain_labels_array = np.hstack(all_domain_labels) if all_domain_labels else None
        input_data_array = np.vstack(all_input_data)

        print(f"domain labels values: {np.unique(domain_labels_array)}" if domain_labels_array is not None else "No domain labels")
        
        return FeatureData(
            features=features_array,
            target_labels=target_labels_array,
            input_data=input_data_array,
            domain_labels=domain_labels_array
        )
    
    def save_features(self, feature_data: FeatureData, output_path: str):
        """특징 데이터 저장"""
        print(f"특징 데이터 저장: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_data = {
            'features': feature_data.features,
            'target_labels': feature_data.target_labels,
            'input_data': feature_data.input_data,
            'subject_name': feature_data.subject_name,
            'model_name': feature_data.model_name,
            'checkpoint_path': feature_data.checkpoint_path
        }
        
        if feature_data.domain_labels is not None:
            save_data['domain_labels'] = feature_data.domain_labels
        
        # 메타데이터 추가
        for key, value in feature_data.metadata.items():
            save_data[key] = value
        
        np.savez_compressed(output_path, **save_data)
    
    def extract_from_checkpoint_info(
        self, 
        checkpoint_info: pd.Series,
        test_config_base_path: str
    ) -> Optional[FeatureData]:
        """체크포인트 정보로부터 특징 추출"""
        subject_name = checkpoint_info['test_subject_name']
        checkpoint_path = checkpoint_info['checkpoint_path']
        
        # 테스트 설정 파일 찾기
        test_config_files = glob.glob(
            os.path.join(test_config_base_path, subject_name, "**", "*.json"),
            recursive=True
        )
        
        if not test_config_files:
            print(f"테스트 설정 파일 없음: {subject_name}")
            return None
        
        try:
            # 모델 정보 추출
            model_info = self._extract_model_info(checkpoint_path)
            
            # 모델 로드
            model = self.load_model_from_checkpoint(
                checkpoint_path, 
                model_info.get('model_name', 'EEGNet')
            )
            
            # 데이터 모듈 생성
            aug_config = None
            if self.config.use_augmentation and self.config.augmentation_config_path:
                aug_config = self.create_augmentation_config(self.config.augmentation_config_path)
            
            data_module = DataModuleFactory.create_data_module(
                test_config_files[0],
                model_info.get('batch_size', self.config.batch_size),
                self.config.test_data_path,
                aug_config
            )
            
            # 특징 추출
            feature_data = self.extract_features(model, data_module)
            
            # 메타데이터 설정
            feature_data.subject_name = subject_name
            feature_data.model_name = model_info.get('model_name', 'EEGNet')
            feature_data.checkpoint_path = checkpoint_path
            feature_data.metadata.update({
                'config_path': test_config_files[0],
                'augmentation_enabled': aug_config is not None and aug_config.get('enabled', False)
            })
            
            return feature_data
            
        except Exception as e:
            print(f"특징 추출 실패 ({subject_name}): {e}")
            return None
    
    def _extract_model_info(self, experiment_path: str) -> Dict:
        """실험 경로에서 모델 정보 추출"""
        yaml_files = glob.glob(os.path.join(os.path.dirname(experiment_path), '*.yml'))
        
        if not yaml_files:
            return {'model_name': 'EEGNet', 'batch_size': self.config.batch_size}
        
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            
            search_space = config.get('search_space', {})
            return {
                'model_name': search_space.get('model_name', 'EEGNet'),
                'batch_size': search_space.get('batch_size', self.config.batch_size)
            }
        except:
            return {'model_name': 'EEGNet', 'batch_size': self.config.batch_size}