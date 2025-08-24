"""
RayTune 체크포인트에서 특징 추출 및 저장을 수행하는 모듈

주요 기능:
1. checkpoint_analysis_results.csv에서 체크포인트 경로 로드
2. 체크포인트로부터 모델 복원
3. 테스트 데이터에서 특징 벡터 추출 및 저장
4. 추출된 특징, 타겟 레이블, 도메인 레이블, 입력 데이터 등을 NPZ 파일로 저장
반환 타입이 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]로 변경 (입력 데이터 추가)
출력 정보 개선:
입력 데이터 shape 정보도 출력하도록 추가
이제 NPZ 파일에는 다음 데이터들이 저장:
    features: 추출된 특징 벡터
    target_labels: 타겟 레이블
    domain_labels: 도메인 레이블 (있는 경우)
    input_data: 원본 입력 데이터
    subject_name: 피험자 이름
    model_name: 모델 이름
    checkpoint_path: 체크포인트 경로
"""
import os
import sys
import json
import glob
import yaml
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 모델 및 데이터 모듈 import
from src.models.eegnet import EEGNet
from src.models.eegnet_grl import EEGNetGRL, EEGNetLNL, EEGNetMI, EEGNetLNLAutoCorrelation
from src.models.eegnetDRO import EEGNetDRO
from src.models.eegnet_grl_lag import EEGNetLNLLag
from src.data.modules.EEGdataModuel import EEGDataModule


class FeatureExtractor:
    """체크포인트 기반 특징 추출 클래스"""
    
    def __init__(self, test_data_default_path: str):
        self.test_data_default_path = test_data_default_path
        self.model_dict = {
            'EEGNet': EEGNet,
            'EEGNetDomainAdaptation_LNL': EEGNetLNL,
            'EEGNetDomainAdaptation_Not_GRL': EEGNetMI,
            'EEGNetDomainAdaptation_Only_GRL': EEGNetGRL,
            'EEGNetLNL': EEGNetLNL,
            'EEGNetMI': EEGNetMI,
            'EEGNetGRL': EEGNetGRL,
            'EEGNetLNLLag': EEGNetLNLLag,
            'EEGNetLNLAutoCorrelation': EEGNetLNLAutoCorrelation,
            'EEGNetDRO': EEGNetDRO
        }
    
    def _load_cortical_regions(self, regions_path: str):
        """
        cortical_regions.json 파일을 로드하여 반환
        :param regions_path: regions.json 파일 경로
        :return: List[List[int]] 형태의 cortical regions
        """
        if regions_path is None:
            raise ValueError("Cortical regions path is required for data augmentation.")
        
        with open(regions_path, 'r') as f:
            regions = json.load(f)
        
        regions_list = []
        for key, values in regions.items():
            regions_list.append(values)

        return regions_list

    def _prepare_augmentation_config_for_datamodule(self, da_config: dict):
        """
        EEGDataModule에 전달할 데이터 증강 설정을 최종적으로 구성합니다.
        YAML에서 로드된 설정과, 필요한 경우 추가적으로 처리된 데이터를 조합합니다.

        :param da_config: YAML 파일에서 로드된 데이터 증강 설정
        :return: EEGDataModule의 data_augmentation_config 파라미터에 전달될 최종 설정 딕셔너리
        """
        final_config = {
            'enabled': da_config.get('enabled', True),
            'train_only': da_config.get('train_only', True),
            'methods': []
        }

        # 설정된 메서드 목록을 순회하며 필요한 정보를 구성합니다.
        for method_name in da_config.get('methods', []):
            method_setting = da_config.get('setting', {}).get(method_name)
            if not method_setting:
                print(f"Warning: Augmentation method '{method_name}' has no setting. Skipping.")
                continue

            method_info = {
                'type': method_setting.get('name'), # 'cortical' 또는 'subject'
                'prob_method': method_setting.get('swap_probability_method', 'uniform')
            }

            # CorticalRegionChannelSwap의 경우, regions 정보가 필요합니다.
            if method_name == 'CorticalRegionChannelSwap':
                regions_path = method_setting.get('cortical_regions_path')
                if not regions_path:
                    raise ValueError("cortical_regions_path is required for CorticalRegionChannelSwap.")
                
                # 실제 경로는 default_path와 결합하여 사용합니다.
                full_regions_path = os.path.join(self.test_data_default_path, regions_path)
                method_info['regions'] = self._load_cortical_regions(full_regions_path)
            
            # SubjectLevelChannelSwap의 경우, enable_soft_labels 정보가 필요합니다.
            elif method_name == 'SubjectLevelChannelSwap':
                method_info['enable_soft_labels'] = method_setting.get('enable_soft_labels', False)

            final_config['methods'].append(method_info)
            
        return final_config

    def create_data_augmentation_config(self, da_config_path: str = None):
        """
        데이터 증강 설정을 위한 config 생성
        1. da_config_path에 지정된 YAML 파일을 로드합니다.
        2. 로드된 설정을 EEGDataModule에 적합한 형태로 변환합니다.
        
        :param da_config_path: 데이터 증강 설정 YAML 파일 경로
        :return: EEGDataModule에 전달할 데이터 증강 설정 딕셔너리
        """
        if not da_config_path:
            return {'enabled': False}  # 데이터 증강 비활성화
        
        full_config_path = os.path.join(self.test_data_default_path, da_config_path)
        print(f"📋 데이터 증강 설정 로드: {full_config_path}")
        
        try:
            with open(full_config_path, 'r') as f:
                loaded_da_config = yaml.safe_load(f)
            
            # YAML에서 로드된 설정을 실제 값으로 변환 (Ray Tune 설정 스타일 처리)
            mapped_da_config = {}
            for key, value in loaded_da_config.items():
                if isinstance(value, dict) and 'value' in value:
                    mapped_da_config[key] = value['value']
                else:
                    mapped_da_config[key] = value
            
            # EEGDataModule에 전달하기 위해 최종적으로 설정 포맷팅
            final_da_config = self._prepare_augmentation_config_for_datamodule(mapped_da_config)
            final_da_config['enabled'] = True  # 최종적으로 증강 활성화 상태 명시

            print(f"✅ 데이터 증강 설정 완료: {final_da_config}")
            return final_da_config
            
        except FileNotFoundError:
            print(f"❌ 데이터 증강 설정 파일을 찾을 수 없습니다: {full_config_path}")
            return {'enabled': False}
        except Exception as e:
            print(f"❌ 데이터 증강 설정 로드 실패: {e}")
            return {'enabled': False}
    
    def load_model_from_checkpoint(self, checkpoint_path: str, model_name: str):
        """체크포인트에서 모델 로드"""
        model_class = self.model_dict.get(model_name)
        if not model_class:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.ckpt")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {ckpt_file}")
        
        print(f"📦 체크포인트에서 모델 로드: {ckpt_file}")
        model = model_class.load_from_checkpoint(ckpt_file)
        model.eval()
        return model
    
    def create_data_module(self, test_config_path: str, batch_size: int = 16, data_augmentation_config: dict = None):
        """테스트 데이터 모듈 생성"""
        # 경로 보정
        config_path = test_config_path
        old_base = "/mnt/sdb1/jsw/hanseul/code/Fairness_for_generalization"
        new_base = "/home/jsw/Fairness/Fairness_for_generalization"
        
        if config_path.startswith(old_base):
            relative_path = os.path.relpath(config_path, old_base)
            config_path = os.path.join(new_base, relative_path)
        
        with open(config_path, 'r') as f:
            data_config = json.load(f)
        
        data_module = EEGDataModule(
            data_config=data_config, batch_size=batch_size, masking_ch_list=[],
            rm_ch_list=[], subject_usage="test1", seed=None,
            default_path=self.test_data_default_path, skip_time_list=None,
            data_augmentation_config=data_augmentation_config
        )
        
        data_module.setup('test')
        print(f"📊 테스트 데이터 로드 완료: {len(data_module.test_dataset)}개 샘플")
        return data_module
    
    def extract_features(self, model, data_module) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """모델에서 특징 추출"""
        model.eval()
        all_features = []
        all_target_labels = []
        all_domain_labels = []
        all_input_data = []
        
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # 배치 구조 확인: (x, target_labels) 또는 (x, target_labels, domain_labels)
                if len(batch) == 2:
                    x, target_labels = batch
                    domain_labels = None
                elif len(batch) == 3:
                    x, target_labels, domain_labels = batch
                else:
                    x = batch[0]
                    target_labels = batch[1]
                    domain_labels = batch[2] if len(batch) > 2 else None
                
                # 입력 데이터 저장 (GPU로 이동 전에 저장)
                all_input_data.append(x.cpu().numpy())
                
                # GPU 사용 가능한 경우 이동
                if torch.cuda.is_available():
                    x = x.cuda()
                    model = model.cuda()
                
                # 특징 추출 (extract_features 메서드 사용)
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                else:
                    # extract_features가 없는 경우 forward로 특징 추출
                    features = model.feature_extractor(x.permute(0, 3, 1, 2))
                    features = torch.flatten(features, start_dim=1)
                
                all_features.append(features.cpu().numpy())
                all_target_labels.append(target_labels.numpy())
                
                # domain_labels가 있는 경우만 추가
                if domain_labels is not None:
                    all_domain_labels.append(domain_labels.numpy())
        
        features_array = np.vstack(all_features)
        target_labels_array = np.hstack(all_target_labels)
        domain_labels_array = np.hstack(all_domain_labels) if all_domain_labels else None
        input_data_array = np.vstack(all_input_data)
        
        print(f"✅ 특징 추출 완료: {features_array.shape}")
        print(f"   Input data: {input_data_array.shape}")
        print(f"   Target labels: {target_labels_array.shape}, 고유값: {np.unique(target_labels_array)}")
        if domain_labels_array is not None:
            print(f"   Domain labels: {domain_labels_array.shape}, 고유값: {np.unique(domain_labels_array)}")
        
        return features_array, target_labels_array, domain_labels_array, input_data_array

    def extract_features_with_predictions(self, model, data_module) -> Dict[str, np.ndarray]:
        """특징 + 예측 결과 동시 추출"""
        model.eval()
        all_features = []
        all_predictions = []
        all_prediction_probs = []
        all_target_labels = []
        all_domain_labels = []
        all_input_data = []
        
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                # 배치 구조 확인
                if len(batch) == 2:
                    x, target_labels = batch
                    domain_labels = None
                elif len(batch) == 3:
                    x, target_labels, domain_labels = batch
                else:
                    x = batch[0]
                    target_labels = batch[1]
                    domain_labels = batch[2] if len(batch) > 2 else None
                
                # 입력 데이터 저장 (GPU로 이동 전에 저장)
                all_input_data.append(x.cpu().numpy())
                
                # GPU 사용 가능한 경우 이동
                if torch.cuda.is_available():
                    x = x.cuda()
                    model = model.cuda()
                
                # 특징 추출
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                else:
                    # extract_features가 없는 경우 feature_extractor 사용
                    features = model.feature_extractor(x.permute(0, 3, 1, 2))
                    features = torch.flatten(features, start_dim=1)
                
                # 예측 수행 (전체 forward pass)
                model_output = model(x)
                # 모델 출력이 tuple인 경우 첫 번째 요소(logits)만 사용
                if isinstance(model_output, tuple):
                    logits = model_output[0]
                else:
                    logits = model_output
                    
                predictions = torch.argmax(logits, dim=1)
                prediction_probs = torch.softmax(logits, dim=1)
                
                # 결과 저장
                all_features.append(features.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_prediction_probs.append(prediction_probs.cpu().numpy())
                all_target_labels.append(target_labels.cpu().numpy())
                
                # domain_labels가 있는 경우만 추가
                if domain_labels is not None:
                    all_domain_labels.append(domain_labels.cpu().numpy())
        
        # 배열 결합
        features_array = np.vstack(all_features)
        predictions_array = np.hstack(all_predictions)
        prediction_probs_array = np.vstack(all_prediction_probs)
        target_labels_array = np.hstack(all_target_labels)
        domain_labels_array = np.hstack(all_domain_labels) if all_domain_labels else None
        input_data_array = np.vstack(all_input_data)
        
        # 정확도 계산
        correct_predictions = predictions_array == target_labels_array
        accuracy = np.mean(correct_predictions)
        
        # 클래스별 정확도 계산
        per_class_accuracy = {}
        for class_id in np.unique(target_labels_array):
            class_mask = target_labels_array == class_id
            if np.sum(class_mask) > 0:
                per_class_accuracy[int(class_id)] = np.mean(correct_predictions[class_mask])
        
        print(f"✅ 특징 + 예측 추출 완료: {features_array.shape}")
        print(f"   전체 정확도: {accuracy:.4f}")
        print(f"   클래스별 정확도: {per_class_accuracy}")
        print(f"   예측 분포: {np.bincount(predictions_array)}")
        print(f"   실제 분포: {np.bincount(target_labels_array)}")
        
        return {
            'features': features_array,
            'predictions': predictions_array,
            'prediction_probs': prediction_probs_array,
            'target_labels': target_labels_array,
            'domain_labels': domain_labels_array,
            'input_data': input_data_array,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy
        }

    def extract_augmented_features(self, model, data_module_original, data_module_augmented) -> Dict:
        """원본과 증강된 데이터에서 특징 추출"""
        print("🔄 원본 데이터 특징 추출")
        orig_features, orig_target_labels, orig_domain_labels, orig_input_data = self.extract_features(model, data_module_original)
        
        print("🔄 증강된 데이터 특징 추출")
        aug_features, aug_target_labels, aug_domain_labels, aug_input_data = self.extract_features(model, data_module_augmented)
        
        return {
            'original': {
                'features': orig_features,
                'target_labels': orig_target_labels,
                'domain_labels': orig_domain_labels,
                'input_data': orig_input_data
            },
            'augmented': {
                'features': aug_features,
                'target_labels': aug_target_labels,
                'domain_labels': aug_domain_labels,
                'input_data': aug_input_data
            }
        }


class CheckpointFeatureRunner:
    """체크포인트 분석 결과 기반 특징 추출 실행기"""
    
    def __init__(self, analysis_result_path: str, test_config_base_path: str, test_data_default_path: str):
        self.analysis_result_path = analysis_result_path
        self.test_config_base_path = test_config_base_path
        self.extractor = FeatureExtractor(test_data_default_path)
    
    def load_analysis_results(self) -> pd.DataFrame:
        """분석 결과 로드"""
        df = pd.read_csv(self.analysis_result_path)
        valid_df = df[df['checkpoint_found'] == True].copy()
        print(f"📋 유효한 체크포인트: {len(valid_df)}개")
        return valid_df
    
    def get_test_config_files(self, subject_name: str) -> List[str]:
        """테스트 설정 파일 찾기"""
        subject_path = os.path.join(self.test_config_base_path, subject_name)
        if not os.path.exists(subject_path):
            return []
        return glob.glob(os.path.join(subject_path, '**', '*.json'), recursive=True)
    
    def extract_model_info(self, experiment_path: str) -> Dict:
        """모델 정보 추출"""
        yaml_files = glob.glob(os.path.join(os.path.dirname(experiment_path), '*.yml'))
        if not yaml_files:
            return {'model_name': 'EEGNetLNL', 'batch_size': 16}
        
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            search_space = config.get('search_space', {})
            return {
                'model_name': search_space.get('model_name', 'EEGNetLNL'),
                'batch_size': search_space.get('batch_size', 16)
            }
        except:
            return {'model_name': 'EEGNetLNL', 'batch_size': 16}
    
    def run_single_extraction(self, checkpoint_info: pd.Series, output_dir: str, 
                             data_augmentation_config_path: str = None,
                             include_predictions: bool = True) -> bool:
        """단일 체크포인트 특징 추출"""
        subject_name = checkpoint_info['test_subject_name']
        checkpoint_path = checkpoint_info['checkpoint_path']
        experiment_path = checkpoint_info['experiment_path']
        
        print(f"\n🎯 특징 추출 시작: {subject_name}")
        
        # 모델 정보 및 설정 파일
        model_info = self.extract_model_info(experiment_path)
        test_config_files = self.get_test_config_files(subject_name)
        
        if not test_config_files:
            print(f"❌ {subject_name} 테스트 설정 없음")
            return False
        
        try:
            # 모델 로드
            model = self.extractor.load_model_from_checkpoint(checkpoint_path, model_info['model_name'])
            
            # 데이터 증강 설정 생성 (선택사항)
            data_augmentation_config = None
            if data_augmentation_config_path:
                data_augmentation_config = self.extractor.create_data_augmentation_config(data_augmentation_config_path)
            
            # 데이터 모듈 생성
            data_module = self.extractor.create_data_module(
                test_config_files[0], 
                model_info['batch_size'],
                data_augmentation_config
            )
            
            # 특징 추출 (예측 포함 여부에 따라 분기)
            if include_predictions:
                print(f"🔍 예측 정보 포함 특징 추출 중...")
                extraction_result = self.extractor.extract_features_with_predictions(model, data_module)
                
                # 저장할 데이터 구성 (예측 정보 포함)
                save_data = {
                    'features': extraction_result['features'],
                    'target_labels': extraction_result['target_labels'],
                    'predictions': extraction_result['predictions'],
                    'prediction_probs': extraction_result['prediction_probs'],
                    'correct_predictions': extraction_result['correct_predictions'],
                    'input_data': extraction_result['input_data'],
                    'accuracy': extraction_result['accuracy'],
                    'per_class_accuracy': str(extraction_result['per_class_accuracy']),
                    'subject_name': subject_name,
                    'model_name': model_info['model_name'],
                    'checkpoint_path': checkpoint_path
                }
                
                # domain_labels가 있는 경우만 추가
                if extraction_result['domain_labels'] is not None:
                    save_data['domain_labels'] = extraction_result['domain_labels']
                
                output_filename = f"{subject_name}_features_with_predictions.npz"
                
            else:
                print(f"🔍 기본 특징 추출 중...")
                features, target_labels, domain_labels, input_data = self.extractor.extract_features(model, data_module)
                
                # 저장할 데이터 구성 (기존 방식)
                save_data = {
                    'features': features, 
                    'target_labels': target_labels,
                    'input_data': input_data,
                    'subject_name': subject_name,
                    'model_name': model_info['model_name'],
                    'checkpoint_path': checkpoint_path
                }
                
                # domain_labels가 있는 경우만 추가
                if domain_labels is not None:
                    save_data['domain_labels'] = domain_labels
                
                output_filename = f"{subject_name}_features.npz"
            
            # 저장
            output_file = os.path.join(output_dir, output_filename)
            os.makedirs(output_dir, exist_ok=True)
            
            # 데이터 증강 정보 저장
            if data_augmentation_config and data_augmentation_config.get('enabled', False):
                save_data['data_augmentation_enabled'] = True
                save_data['data_augmentation_config'] = str(data_augmentation_config)
            
            np.savez_compressed(output_file, **save_data)
            
            if include_predictions:
                print(f"💾 예측 포함 특징 저장: {output_file}")
                print(f"   정확도: {extraction_result['accuracy']:.4f}")
            else:
                print(f"💾 특징 저장: {output_file} (shape: {features.shape})")
            
            return True
            
        except Exception as e:
            print(f"❌ 특징 추출 실패: {e}")
            return False
    
    def run_all_extractions(self, output_base_dir: str, max_extractions: Optional[int] = None,
                           data_augmentation_config_path: str = None,
                           include_predictions: bool = True) -> Dict:
        """모든 체크포인트 특징 추출"""
        print("🚀 특징 추출 시작\n")
        
        if data_augmentation_config_path:
            print(f"📋 데이터 증강 설정 사용: {data_augmentation_config_path}")
        
        if include_predictions:
            print(f"🔍 예측 정보 포함 추출 모드")
        else:
            print(f"🔍 기본 특징 추출 모드")
        
        valid_checkpoints = self.load_analysis_results()
        if max_extractions:
            valid_checkpoints = valid_checkpoints.head(max_extractions)
        
        results = {'success': [], 'failed': []}
        
        for idx, checkpoint_info in valid_checkpoints.iterrows():
            subject_name = checkpoint_info['test_subject_name']
            output_dir = os.path.join(output_base_dir, subject_name)
            
            if self.run_single_extraction(checkpoint_info, output_dir, 
                                        data_augmentation_config_path, include_predictions):
                results['success'].append(subject_name)
            else:
                results['failed'].append(subject_name)
        
        print(f"\n📊 특징 추출 완료: 성공 {len(results['success'])}개, 실패 {len(results['failed'])}개")
        return results


def extract_checkpoint_features(analysis_result_path: str, test_config_base_path: str, 
                               test_data_default_path: str, output_dir: str, 
                               max_extractions: Optional[int] = None,
                               data_augmentation_config_path: str = None,
                               include_predictions: bool = True) -> Dict:
    """체크포인트 특징 추출 실행 함수"""
    runner = CheckpointFeatureRunner(
        analysis_result_path=analysis_result_path,
        test_config_base_path=test_config_base_path,
        test_data_default_path=test_data_default_path
    )
    
    return runner.run_all_extractions(output_dir, max_extractions, 
                                    data_augmentation_config_path, include_predictions)


if __name__ == "__main__":
    print("🚀 체크포인트 특징 추출 시스템")
    
    # 기본 설정
    analysis_result_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60/1_inner_wireless/ray_results_test1_Day1,8_finetune_ReduceLROnPlateau_LNL_batch16/analyzing_result/checkpoint_analysis_results.csv"
    test_config_base_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw5&6config/test/only1Day1,8"
    test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    output_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60/1_inner_wireless/ray_results_test1_Day1,8_finetune_ReduceLROnPlateau_LNL_batch16/analyzing_result/extracted_features"
    
    # 데이터 증강 설정 (선택사항)
    data_augmentation_config_path = "config/RayTune/raw5&6/data_augmentation/data_augmentation.yml"
    
    try:
        results = extract_checkpoint_features(
            analysis_result_path=analysis_result_path,
            test_config_base_path=test_config_base_path,
            test_data_default_path=test_data_default_path,
            output_dir=output_dir,
            max_extractions=3,
            data_augmentation_config_path=data_augmentation_config_path
        )
        print(f"✅ 특징 추출 완료: {results}")
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()
