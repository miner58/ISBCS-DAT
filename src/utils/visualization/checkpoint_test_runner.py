"""
RayTune 체크포인트에서 특징 추출 및 저장을 수행하는 모듈

주요 기능:
1. checkpoint_analysis_results.csv에서 체크포인트 경로 로드
2. 체크포인트로부터 모델 복원
3. 테스트 데이터에서 특징 벡터 추출 및 저장
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

# 모델 및 데이터 모듈 import
from src.models import EEGNet
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
    
    def create_data_module(self, test_config_path: str, batch_size: int = 16):
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
            default_path=self.test_data_default_path, skip_time_list=None
        )
        
        data_module.setup('test')
        print(f"📊 테스트 데이터 로드 완료: {len(data_module.test_dataset)}개 샘플")
        return data_module
    
    def extract_features(self, model, data_module) -> Tuple[np.ndarray, np.ndarray]:
        """모델에서 특징 추출"""
        model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                x, labels = batch
                
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
                all_labels.append(labels.numpy())
        
        features_array = np.vstack(all_features)
        labels_array = np.hstack(all_labels)
        
        print(f"✅ 특징 추출 완료: {features_array.shape}")
        return features_array, labels_array


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
    
    def run_single_extraction(self, checkpoint_info: pd.Series, output_dir: str) -> bool:
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
            # 모델 로드 및 데이터 준비
            model = self.extractor.load_model_from_checkpoint(checkpoint_path, model_info['model_name'])
            data_module = self.extractor.create_data_module(test_config_files[0], model_info['batch_size'])
            
            # 특징 추출
            features, labels = self.extractor.extract_features(model, data_module)
            
            # 저장
            output_file = os.path.join(output_dir, f"{subject_name}_features.npz")
            os.makedirs(output_dir, exist_ok=True)
            
            np.savez_compressed(output_file, 
                              features=features, 
                              labels=labels,
                              subject_name=subject_name,
                              model_name=model_info['model_name'],
                              checkpoint_path=checkpoint_path)
            
            print(f"💾 특징 저장: {output_file} (shape: {features.shape})")
            return True
            
        except Exception as e:
            print(f"❌ 특징 추출 실패: {e}")
            return False
    
    def run_all_extractions(self, output_base_dir: str, max_extractions: Optional[int] = None) -> Dict:
        """모든 체크포인트 특징 추출"""
        print("🚀 특징 추출 시작\n")
        
        valid_checkpoints = self.load_analysis_results()
        if max_extractions:
            valid_checkpoints = valid_checkpoints.head(max_extractions)
        
        results = {'success': [], 'failed': []}
        
        for idx, checkpoint_info in valid_checkpoints.iterrows():
            subject_name = checkpoint_info['test_subject_name']
            output_dir = os.path.join(output_base_dir, subject_name)
            
            if self.run_single_extraction(checkpoint_info, output_dir):
                results['success'].append(subject_name)
            else:
                results['failed'].append(subject_name)
        
        print(f"\n📊 특징 추출 완료: 성공 {len(results['success'])}개, 실패 {len(results['failed'])}개")
        return results


def extract_checkpoint_features(analysis_result_path: str, test_config_base_path: str, 
                               test_data_default_path: str, output_dir: str, 
                               max_extractions: Optional[int] = None) -> Dict:
    """체크포인트 특징 추출 실행 함수"""
    runner = CheckpointFeatureRunner(
        analysis_result_path=analysis_result_path,
        test_config_base_path=test_config_base_path,
        test_data_default_path=test_data_default_path
    )
    
    return runner.run_all_extractions(output_dir, max_extractions)


if __name__ == "__main__":
    print("🚀 체크포인트 특징 추출 시스템")
    
    # 기본 설정
    analysis_result_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/analyzing_result/checkpoint_analysis_results.csv"
    test_config_base_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw3config/test/only1Day1,8"
    test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/preprocessed/at_least"
    output_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/extracted_features"
    
    try:
        results = extract_checkpoint_features(
            analysis_result_path=analysis_result_path,
            test_config_base_path=test_config_base_path,
            test_data_default_path=test_data_default_path,
            output_dir=output_dir,
            max_extractions=3
        )
        print(f"✅ 특징 추출 완료: {results}")
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()
    
    def load_analysis_results(self) -> pd.DataFrame:
        """체크포인트 분석 결과 로드"""
        if not os.path.exists(self.analysis_result_path):
            raise FileNotFoundError(f"분석 결과 파일이 없습니다: {self.analysis_result_path}")
        
        df = pd.read_csv(self.analysis_result_path)
        print(f"📋 분석 결과 로드: {len(df)}개 체크포인트")
        
        # 유효한 체크포인트만 필터링
        valid_df = df[df['checkpoint_found'] == True].copy()
        print(f"✅ 유효한 체크포인트: {len(valid_df)}개")
        
        return valid_df
    
    def get_test_config_files(self, subject_name: str) -> List[str]:
        """주체별 테스트 설정 파일 찾기"""
        subject_path = os.path.join(self.test_config_base_path, subject_name)
        if not os.path.exists(subject_path):
            print(f"⚠️ 주체 폴더가 없습니다: {subject_path}")
            return []
        
        # 재귀적으로 JSON 파일 찾기
        config_files = glob.glob(os.path.join(subject_path, '**', '*.json'), recursive=True)
        print(f"📁 {subject_name} 테스트 설정 파일: {len(config_files)}개")
        
        return config_files
    
    def extract_model_info_from_path(self, experiment_path: str) -> Dict:
        """실험 경로에서 모델 정보 추출"""
        # 실험 폴더에서 YAML 설정 파일 찾기
        yaml_files = glob.glob(os.path.join(os.path.dirname(experiment_path), '*.yml'))
        if not yaml_files:
            # 기본값 반환
            return {
                'model_name': 'EEGNetLNL',  # 일반적으로 사용되는 모델
                'batch_size': 16
            }
        
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            
            # search_space에서 모델 정보 추출
            search_space = config.get('search_space', {})
            return {
                'model_name': search_space.get('model_name', 'EEGNetLNL'),
                'batch_size': search_space.get('batch_size', 16)
            }
        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패: {e}")
            return {
                'model_name': 'EEGNetLNL',
                'batch_size': 16
            }
    
    def run_single_test(self, checkpoint_info: pd.Series) -> Dict:
        """단일 체크포인트 테스트"""
        subject_name = checkpoint_info['test_subject_name']
        checkpoint_path = checkpoint_info['checkpoint_path']
        experiment_path = checkpoint_info['experiment_path']
        
        print(f"\n🎯 테스트 시작: {subject_name}")
        print(f"   체크포인트: {os.path.basename(checkpoint_path)}")
        
        # 모델 정보 추출
        model_info = self.extract_model_info_from_path(experiment_path)
        model_name = model_info['model_name']
        batch_size = model_info['batch_size']
        
        # 테스트 설정 파일 찾기
        test_config_files = self.get_test_config_files(subject_name)
        if not test_config_files:
            print(f"❌ {subject_name}의 테스트 설정 파일을 찾을 수 없습니다.")
            return {}
        
        # 모델 로드
        try:
            model = self.tester.load_model_from_checkpoint(checkpoint_path, model_name)
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return {}
        
        # 각 테스트 설정에 대해 테스트 실행
        all_results = []
        for config_file in test_config_files[:1]:  # 첫 번째 설정만 사용 (시간 단축)
            try:
                data_module = self.tester.create_data_module(config_file, batch_size)
                test_results = self.tester.run_test(model, data_module)
                
                # 결과에 메타 정보 추가
                test_results.update({
                    'subject_name': subject_name,
                    'test_config': os.path.basename(config_file),
                    'model_name': model_name,
                    'checkpoint_path': checkpoint_path,
                    'grl_lambda': checkpoint_info['grl_lambda'],
                    'lnl_lambda': checkpoint_info['lnl_lambda']
                })
                
                all_results.append(test_results)
                
            except Exception as e:
                print(f"❌ 테스트 실행 실패: {e}")
                continue
        
        return all_results[0] if all_results else {}
    
    def run_all_tests(self, max_tests: Optional[int] = None) -> pd.DataFrame:
        """모든 체크포인트에 대해 테스트 실행"""
        print("🚀 체크포인트 테스트 시작\n")
        
        # 분석 결과 로드
        valid_checkpoints = self.load_analysis_results()
        
        if max_tests:
            valid_checkpoints = valid_checkpoints.head(max_tests)
            print(f"🔢 테스트 제한: {max_tests}개")
        
        # 각 체크포인트에 대해 테스트 실행
        test_results = []
        for idx, checkpoint_info in valid_checkpoints.iterrows():
            try:
                result = self.run_single_test(checkpoint_info)
                if result:
                    test_results.append(result)
                    print(f"✅ {checkpoint_info['test_subject_name']} 테스트 완료")
                else:
                    print(f"❌ {checkpoint_info['test_subject_name']} 테스트 실패")
            except Exception as e:
                print(f"❌ 테스트 중 오류: {e}")
                continue
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(test_results)
        
        print(f"\n📊 전체 테스트 완료: {len(results_df)}/{len(valid_checkpoints)}개 성공")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """결과 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"💾 결과 저장: {output_path}")


def run_checkpoint_tests(analysis_result_path: str, test_config_base_path: str, 
                        test_data_default_path: str, output_path: str, 
                        max_tests: Optional[int] = None):
    """
    체크포인트 테스트 실행 함수
    
    Args:
        analysis_result_path: checkpoint_analysis_results.csv 파일 경로
        test_config_base_path: 테스트 설정 파일 기본 경로
        test_data_default_path: 테스트 데이터 기본 경로
        output_path: 결과 저장 경로
        max_tests: 최대 테스트 수 (None이면 모든 체크포인트 테스트)
    """
    runner = CheckpointFeatureRunner(
        analysis_result_path=analysis_result_path,
        test_config_base_path=test_config_base_path,
        test_data_default_path=test_data_default_path
    )
    
    # 테스트 실행
    results_df = runner.run_all_tests(max_tests=max_tests)
    
    # 결과 저장
    if not results_df.empty:
        runner.save_results(results_df, output_path)
        
        # 간단한 통계 출력
        print(f"\n📈 테스트 결과 통계:")
        if 'test/report/macro avg/accuracy' in results_df.columns:
            acc_col = 'test/report/macro avg/accuracy'
            print(f"   평균 정확도: {results_df[acc_col].mean():.4f}")
            print(f"   최고 정확도: {results_df[acc_col].max():.4f}")
            print(f"   최저 정확도: {results_df[acc_col].min():.4f}")
    else:
        print("❌ 테스트 결과가 없습니다.")
    
    return results_df


def run_batch_checkpoint_tests(base_experiment_paths: List[str], 
                              test_config_mapping: Dict[str, str],
                              test_data_default_path: str,
                              max_tests_per_experiment: Optional[int] = None):
    """
    여러 실험에 대해 일괄 체크포인트 테스트 실행
    
    Args:
        base_experiment_paths: 실험 기본 경로 리스트
        test_config_mapping: 데이터 타입별 테스트 설정 경로 매핑
        test_data_default_path: 테스트 데이터 기본 경로
        max_tests_per_experiment: 실험당 최대 테스트 수
    
    Example:
        test_config_mapping = {
            'wire': '/path/to/raw3config/test/only1Day1,8',
            'wireless': '/path/to/raw5&6config/test/only1Day1,8'
        }
    """
    all_results = []
    
    for base_path in base_experiment_paths:
        print(f"\n🎯 실험 기본 경로 처리: {base_path}")
        
        # 실험 타입별로 처리
        for data_type, test_config_path in test_config_mapping.items():
            print(f"📊 데이터 타입: {data_type}")
            
            # 해당 타입의 실험 폴더 찾기
            experiment_folders = []
            if data_type == "wire":
                patterns = ["*2_inner_wire*", "*3_wireless2wire*"]
            elif data_type == "wireless":
                patterns = ["*1_inner_wireless*", "*4_wire2wireless*"]
            else:
                print(f"⚠️ 알 수 없는 데이터 타입: {data_type}")
                continue
            
            # 패턴에 맞는 폴더 찾기
            for pattern in patterns:
                matching_folders = glob.glob(os.path.join(base_path, pattern))
                experiment_folders.extend(matching_folders)
            
            print(f"📁 {data_type} 실험 폴더: {len(experiment_folders)}개")
            
            # 각 실험 폴더에 대해 테스트 실행
            for exp_folder in experiment_folders:
                ray_results_folders = glob.glob(os.path.join(exp_folder, "ray_results*"))
                
                for ray_folder in ray_results_folders:
                    analysis_result_path = os.path.join(ray_folder, "analyzing_result", "checkpoint_analysis_results.csv")
                    output_path = os.path.join(ray_folder, "analyzing_result", "checkpoint_test_results.csv")
                    
                    if os.path.exists(analysis_result_path):
                        try:
                            print(f"🧪 테스트 실행: {os.path.basename(ray_folder)}")
                            results = run_checkpoint_tests(
                                analysis_result_path=analysis_result_path,
                                test_config_base_path=test_config_path,
                                test_data_default_path=test_data_default_path,
                                output_path=output_path,
                                max_tests=max_tests_per_experiment
                            )
                            
                            if not results.empty:
                                # 실험 메타 정보 추가
                                results['experiment_folder'] = exp_folder
                                results['data_type'] = data_type
                                all_results.append(results)
                                
                        except Exception as e:
                            print(f"❌ 테스트 실패: {e}")
                            continue
                    else:
                        print(f"⚠️ 분석 결과 파일 없음: {analysis_result_path}")
    
    # 전체 결과 통합
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        print(f"\n🎉 전체 배치 테스트 완료:")
        print(f"   총 테스트: {len(combined_results)}개")
        print(f"   실험 폴더: {combined_results['experiment_folder'].nunique()}개")
        
        # 전체 결과 저장
        batch_output_path = "/tmp/batch_checkpoint_test_results.csv"
        combined_results.to_csv(batch_output_path, index=False)
        print(f"💾 배치 결과 저장: {batch_output_path}")
        
        return combined_results
    else:
        print("❌ 배치 테스트 결과가 없습니다.")
        return pd.DataFrame()


if __name__ == "__main__":
    # 예시 실행
    analysis_result_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60/1_inner_wireless/ray_results_test1_Day1,8_finetune_ReduceLROnPlateau_LNL_batch16/analyzing_result/checkpoint_analysis_results.csv"
    test_config_base_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw5&6config/test/only1Day1,8"
    test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    output_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60/1_inner_wireless/ray_results_test1_Day1,8_finetune_ReduceLROnPlateau_LNL_batch16/analyzing_result/checkpoint_test_results.csv"
    
    try:
        results = run_checkpoint_tests(
            analysis_result_path=analysis_result_path,
            test_config_base_path=test_config_base_path,
            test_data_default_path=test_data_default_path,
            output_path=output_path,
            max_tests=3  # 테스트를 위해 3개만 실행
        )
        print("✅ 체크포인트 테스트 완료")
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        import traceback
        traceback.print_exc()
