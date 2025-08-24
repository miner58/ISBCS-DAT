"""
Analysis Pipeline Module

체크포인트 분석을 위한 통합 파이프라인
"""

import os
import glob
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .feature_extractor import FeatureExtractor, ExtractionConfig, FeatureData
from .tsne_visualizer import TSNEVisualizer, VisualizationConfig


@dataclass
class SubjectConfig:
    """피험자별 설정"""
    subject_name: str
    checkpoint_path: str
    test_config_path: str
    model_name: str = "EEGNetLNL"


@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_name: str
    analysis_result_path: str
    test_config_base_path: str
    output_dir: str
    max_subjects: Optional[int] = None


@dataclass 
class PipelineConfig:
    """파이프라인 통합 설정"""
    extraction: ExtractionConfig
    visualization: VisualizationConfig
    save_intermediate: bool = True
    parallel_processing: bool = False


class AnalysisPipeline:
    """체크포인트 분석 통합 파이프라인"""
    
    def __init__(
        self,
        extraction_config: ExtractionConfig,
        visualization_config: VisualizationConfig = None
    ):
        self.extraction_config = extraction_config
        self.visualization_config = visualization_config or VisualizationConfig()
        
        self.extractor = FeatureExtractor(extraction_config)
        self.visualizer = TSNEVisualizer(self.visualization_config)
        
        self._results_cache = {}
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'AnalysisPipeline':
        """설정 파일에서 파이프라인 생성"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        extraction_config = ExtractionConfig(**config_data.get('extraction', {}))
        visualization_config = VisualizationConfig(**config_data.get('visualization', {}))
        
        return cls(extraction_config, visualization_config)
    
    def run_single_subject_analysis(
        self,
        subject_config: SubjectConfig,
        output_dir: str,
        create_plots: bool = True
    ) -> Dict[str, Any]:
        """단일 피험자 분석"""
        print(f"🎯 피험자 분석: {subject_config.subject_name}")
        
        try:
            # 모델 로드
            model = self.extractor.load_model_from_checkpoint(
                subject_config.checkpoint_path,
                subject_config.model_name
            )
            
            # 데이터 모듈 생성
            aug_config = None
            if self.extraction_config.use_augmentation:
                aug_config = self.extractor.create_augmentation_config(
                    self.extraction_config.augmentation_config_path
                )
            
            from .feature_extractor import DataModuleFactory
            data_module = DataModuleFactory.create_data_module(
                subject_config.test_config_path,
                self.extraction_config.batch_size,
                self.extraction_config.test_data_path,
                aug_config
            )
            
            # 특징 추출
            feature_data = self.extractor.extract_features(model, data_module)
            feature_data.subject_name = subject_config.subject_name
            feature_data.model_name = subject_config.model_name
            feature_data.checkpoint_path = subject_config.checkpoint_path
            
            # 결과 저장
            output_path = os.path.join(output_dir, f"{subject_config.subject_name}_features.npz")
            self.extractor.save_features(feature_data, output_path)
            
            # 시각화 생성
            if create_plots:
                plot_output_dir = os.path.join(output_dir, "plots")
                self.visualizer.create_input_vs_feature_plot(feature_data, plot_output_dir)
            
            return {
                'status': 'success',
                'subject_name': subject_config.subject_name,
                'output_path': output_path,
                'feature_shape': feature_data.features.shape,
                'input_shape': feature_data.input_data.shape
            }
            
        except Exception as e:
            print(f"❌ {subject_config.subject_name} 분석 실패: {e}")
            return {
                'status': 'failed',
                'subject_name': subject_config.subject_name,
                'error': str(e)
            }
    
    def run_experiment_analysis(
        self,
        experiment_config: ExperimentConfig,
        create_combined_plots: bool = True
    ) -> Dict[str, Any]:
        """실험 전체 분석"""
        print(f"🚀 실험 분석: {experiment_config.experiment_name}")
        
        # 체크포인트 분석 결과 로드
        if not os.path.exists(experiment_config.analysis_result_path):
            raise FileNotFoundError(f"분석 결과 파일 없음: {experiment_config.analysis_result_path}")
        
        df = pd.read_csv(experiment_config.analysis_result_path)
        valid_checkpoints = df[df['checkpoint_found'] == True].copy()
        
        if experiment_config.max_subjects:
            valid_checkpoints = valid_checkpoints.head(experiment_config.max_subjects)
        
        print(f"📊 처리할 체크포인트: {len(valid_checkpoints)}개")
        
        # 피험자별 분석
        results = []
        feature_data_list = []
        
        for _, checkpoint_info in valid_checkpoints.iterrows():
            subject_name = checkpoint_info['test_subject_name']
            checkpoint_path = checkpoint_info['checkpoint_path']
            
            # 테스트 설정 파일 찾기
            test_config_files = glob.glob(
                os.path.join(experiment_config.test_config_base_path, subject_name, "**", "*.json"),
                recursive=True
            )
            
            if not test_config_files:
                print(f"⚠️ 테스트 설정 없음: {subject_name}")
                continue
            
            # 모델 정보 추출
            model_info = self._extract_model_info_from_checkpoint(checkpoint_path)
            
            subject_config = SubjectConfig(
                subject_name=subject_name,
                checkpoint_path=checkpoint_path,
                test_config_path=test_config_files[0],
                model_name=model_info.get('model_name', 'EEGNetLNL')
            )
            
            # 피험자 분석 실행
            subject_output_dir = os.path.join(experiment_config.output_dir, subject_name)
            result = self.run_single_subject_analysis(
                subject_config,
                subject_output_dir,
                create_plots=True
            )
            
            results.append(result)
            
            # 성공한 경우 특징 데이터 수집 (통합 플롯용)
            if result['status'] == 'success' and create_combined_plots:
                feature_data = self.visualizer.load_feature_data_from_npz(result['output_path'])
                feature_data_list.append(feature_data)
        
        # 통합 플롯 생성
        if create_combined_plots and feature_data_list:
            print("📊 통합 플롯 생성")
            combined_plot_dir = os.path.join(experiment_config.output_dir, "combined_plots")
            self.visualizer.create_combined_subjects_plot(
                feature_data_list,
                experiment_config.experiment_name,
                combined_plot_dir
            )
        
        # 결과 요약
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        return {
            'experiment_name': experiment_config.experiment_name,
            'total_subjects': len(results),
            'success_count': success_count,
            'failed_count': len(results) - success_count,
            'results': results,
            'output_dir': experiment_config.output_dir
        }
    
    def run_augmentation_analysis(
        self,
        subject_configs: Union[SubjectConfig, List[SubjectConfig]],
        output_dir: str,
        plot_types: List[str] = None
    ) -> Dict[str, Any]:
        """데이터 증강 분석 - 단일 또는 다중 피험자 지원"""
        if not self.extraction_config.use_augmentation:
            raise ValueError("데이터 증강이 활성화되지 않음")
        
        plot_types = plot_types or ["type1", "type2", "type3"]
        
        # 1. 입력 정규화
        if isinstance(subject_configs, SubjectConfig):
            subject_configs = [subject_configs]
            is_single_subject = True
        else:
            is_single_subject = False
        
        subject_names = [sc.subject_name for sc in subject_configs]
        print(f"🎨 데이터 증강 분석: {subject_names}")
        
        try:
            # 2. 각 피험자별 특징 추출
            original_feature_list = []
            augmented_feature_list = []
            individual_results = []
            
            for subject_config in subject_configs:
                print(f"  🎯 {subject_config.subject_name} 처리 중...")
                
                # 모델 로드
                model = self.extractor.load_model_from_checkpoint(
                    subject_config.checkpoint_path,
                    subject_config.model_name
                )
                
                from .feature_extractor import DataModuleFactory
                
                # 원본 데이터 모듈 (증강 없음)
                original_data_module = DataModuleFactory.create_data_module(
                    subject_config.test_config_path,
                    self.extraction_config.batch_size,
                    self.extraction_config.test_data_path,
                    None  # 증강 없음
                )
                
                # 증강 데이터 모듈 (증강 있음)
                aug_config = self.extractor.create_augmentation_config(
                    self.extraction_config.augmentation_config_path
                )
                augmented_data_module = DataModuleFactory.create_data_module(
                    subject_config.test_config_path,
                    self.extraction_config.batch_size,
                    self.extraction_config.test_data_path,
                    aug_config
                )
                
                # 특징 추출
                original_data = self.extractor.extract_features(model, original_data_module)
                augmented_data = self.extractor.extract_features(model, augmented_data_module)
                
                # 메타데이터 설정
                original_data.subject_name = subject_config.subject_name
                original_data.model_name = subject_config.model_name
                original_data.checkpoint_path = subject_config.checkpoint_path
                
                augmented_data.subject_name = subject_config.subject_name
                augmented_data.model_name = subject_config.model_name
                augmented_data.checkpoint_path = subject_config.checkpoint_path
                
                original_feature_list.append(original_data)
                augmented_feature_list.append(augmented_data)
                
                # 개별 결과 저장
                individual_results.append({
                    'subject_name': subject_config.subject_name,
                    'original_shape': original_data.features.shape,
                    'augmented_shape': augmented_data.features.shape
                })
                
                print(f"    ✅ {subject_config.subject_name} 특징 추출 완료")
            
            # 3. NPZ 파일 저장
            os.makedirs(output_dir, exist_ok=True)
            original_paths = []
            augmented_paths = []
            
            for i, subject_config in enumerate(subject_configs):
                original_path = os.path.join(output_dir, f"{subject_config.subject_name}_original_features.npz")
                augmented_path = os.path.join(output_dir, f"{subject_config.subject_name}_augmented_features.npz")
                
                self.extractor.save_features(original_feature_list[i], original_path)
                self.extractor.save_features(augmented_feature_list[i], augmented_path)
                
                original_paths.append(original_path)
                augmented_paths.append(augmented_path)
            
            # 4. 통합 시각화 생성
            plot_results = []
            
            if is_single_subject:
                print("  📊 단일 피험자 데이터 증강 플롯 생성")
                # 단일 피험자: 기존 방식
                for plot_type in plot_types:
                    try:
                        fig = self.visualizer.create_augmentation_comparison_plot(
                            original_feature_list[0],
                            augmented_feature_list[0],
                            plot_type,
                            output_dir
                        )
                        plot_results.append(f"{plot_type}_plot_created")
                        print(f"    ✅ {plot_type} 플롯 생성 완료")
                    except Exception as e:
                        print(f"    ⚠️ {plot_type} 플롯 생성 실패: {e}")
                        plot_results.append(f"{plot_type}_plot_failed")
            else:
                print("  📊 다중 피험자 데이터 증강 플롯 생성")
                # 다중 피험자: 새로운 방식
                for plot_type in plot_types:
                    try:
                        # AugmentationComparisonPlot 전략 직접 사용
                        strategy = self.visualizer._strategies['augmentation_comparison']
                        fig = strategy.create_plot(
                            original_feature_list,    # List[FeatureData]
                            augmented_feature_list,   # List[FeatureData]
                            plot_type=plot_type,
                            color_by="subject",       # 피험자별 색상
                            subject_name="MULTI"      # 다중 피험자 모드
                        )
                        
                        # 파일명 생성
                        subject_names_str = "_".join(subject_names)
                        filename = f"multi_subject_augmentation_{plot_type}_{subject_names_str}"
                        strategy.save_plot(fig, output_dir, filename)
                        
                        plot_results.append(f"{plot_type}_plot_created")
                        print(f"    ✅ {plot_type} 다중 피험자 플롯 생성 완료")
                        
                    except Exception as e:
                        print(f"    ⚠️ {plot_type} 다중 피험자 플롯 생성 실패: {e}")
                        plot_results.append(f"{plot_type}_plot_failed")
            
            # 5. 통합 NPZ 파일 저장 (다중 피험자인 경우)
            combined_original_path = None
            combined_augmented_path = None
            
            if not is_single_subject:
                # 여러 피험자 데이터를 하나로 결합
                combined_original = self._combine_feature_data_list(original_feature_list)
                combined_augmented = self._combine_feature_data_list(augmented_feature_list)
                
                combined_original_path = os.path.join(output_dir, "combined_original_features.npz")
                combined_augmented_path = os.path.join(output_dir, "combined_augmented_features.npz")
                
                self.extractor.save_features(combined_original, combined_original_path)
                self.extractor.save_features(combined_augmented, combined_augmented_path)
                
                print(f"    💾 통합 NPZ 파일 저장 완료")
            
            # 6. 결과 반환
            if is_single_subject:
                # 단일 피험자: 기존 호환성
                return {
                    'status': 'success',
                    'subject_name': subject_configs[0].subject_name,
                    'original_path': original_paths[0],
                    'augmented_path': augmented_paths[0],
                    'plots_created': plot_results,
                    'original_shape': original_feature_list[0].features.shape,
                    'augmented_shape': augmented_feature_list[0].features.shape
                }
            else:
                # 다중 피험자: 새로운 구조
                return {
                    'status': 'success',
                    'subject_names': subject_names,
                    'num_subjects': len(subject_configs),
                    'original_paths': original_paths,
                    'augmented_paths': augmented_paths,
                    'combined_original_path': combined_original_path,
                    'combined_augmented_path': combined_augmented_path,
                    'plots_created': plot_results,
                    'individual_results': individual_results,
                    'output_dir': output_dir
                }
            
        except Exception as e:
            print(f"❌ 데이터 증강 분석 실패: {e}")
            if is_single_subject:
                return {
                    'status': 'failed',
                    'subject_name': subject_configs[0].subject_name,
                    'error': str(e)
                }
            else:
                return {
                    'status': 'failed',
                    'subject_names': subject_names,
                    'error': str(e)
                }

    
    def run_augmentation_experiment_analysis(
        self,
        experiment_config: ExperimentConfig,
        plot_types: List[str] = None,
        create_combined_plots: bool = True
    ) -> Dict[str, Any]:
        """데이터 증강 실험 전체 분석"""
        print(f"🎨 데이터 증강 실험 분석: {experiment_config.experiment_name}")
        
        if not self.extraction_config.use_augmentation:
            raise ValueError("데이터 증강이 활성화되지 않음")
        
        plot_types = plot_types or ["type1", "type2", "type3"]
        
        # 체크포인트 분석 결과 로드
        if not os.path.exists(experiment_config.analysis_result_path):
            raise FileNotFoundError(f"분석 결과 파일 없음: {experiment_config.analysis_result_path}")
        
        df = pd.read_csv(experiment_config.analysis_result_path)
        valid_checkpoints = df[df['checkpoint_found'] == True].copy()
        
        if experiment_config.max_subjects:
            valid_checkpoints = valid_checkpoints.head(experiment_config.max_subjects)
        
        print(f"📊 처리할 체크포인트: {len(valid_checkpoints)}개")
        
        # 피험자별 데이터 증강 분석
        results = []
        original_feature_data_list = []
        augmented_feature_data_list = []
        
        for _, checkpoint_info in valid_checkpoints.iterrows():
            subject_name = checkpoint_info['test_subject_name']
            checkpoint_path = checkpoint_info['checkpoint_path']
            
            # 테스트 설정 파일 찾기
            test_config_files = glob.glob(
                os.path.join(experiment_config.test_config_base_path, subject_name, "**", "*.json"),
                recursive=True
            )
            
            if not test_config_files:
                print(f"⚠️ 테스트 설정 없음: {subject_name}")
                continue
            
            # 모델 정보 추출
            model_info = self._extract_model_info_from_checkpoint(checkpoint_path)
            
            subject_config = SubjectConfig(
                subject_name=subject_name,
                checkpoint_path=checkpoint_path,
                test_config_path=test_config_files[0],
                model_name=model_info.get('model_name', 'EEGNetLNL')
            )
            
            # 데이터 증강 분석 실행
            subject_output_dir = os.path.join(experiment_config.output_dir, "augmentation_analysis", subject_name)
            result = self.run_augmentation_analysis(
                subject_config,
                subject_output_dir,
                plot_types
            )
            
            results.append(result)
            
            # 성공한 경우 특징 데이터 수집 (통합 플롯용)
            if result['status'] == 'success' and create_combined_plots:
                original_data = self.visualizer.load_feature_data_from_npz(result['original_path'])
                augmented_data = self.visualizer.load_feature_data_from_npz(result['augmented_path'])
                original_feature_data_list.append(original_data)
                augmented_feature_data_list.append(augmented_data)
        
        # 통합 증강 비교 플롯 생성
        if create_combined_plots and original_feature_data_list and augmented_feature_data_list:
            print("📊 통합 데이터 증강 플롯 생성")
            combined_plot_dir = os.path.join(experiment_config.output_dir, "combined_augmentation_plots")
            
            # 3가지 타입별로 통합 플롯 생성
            for plot_type in plot_types:
                try:
                    # Strategy 패턴을 사용한 증강 비교 플롯
                    strategy = self.visualizer._strategies['augmentation_comparison']
                    fig = strategy.create_plot(
                        original_feature_data_list,
                        augmented_feature_data_list,
                        plot_type=plot_type,
                        color_by="subject",  # 피험자별 색상 구분
                        subject_name="ALL"   # 모든 피험자 통합
                    )
                    
                    # 플롯 저장
                    filename = f"{experiment_config.experiment_name}_combined_augmentation_{plot_type}"
                    strategy.save_plot(fig, combined_plot_dir, filename)
                    
                    print(f"✅ {plot_type} 통합 플롯 저장 완료")
                    
                except Exception as e:
                    print(f"⚠️ {plot_type} 통합 플롯 생성 실패: {e}")
        
        # 결과 요약
        success_count = sum(1 for r in results if r['status'] == 'success')
        
        return {
            'experiment_name': experiment_config.experiment_name,
            'total_subjects': len(results),
            'success_count': success_count,
            'failed_count': len(results) - success_count,
            'plot_types': plot_types,
            'results': results,
            'output_dir': experiment_config.output_dir,
            'combined_plots_dir': os.path.join(experiment_config.output_dir, "combined_augmentation_plots") if create_combined_plots else None
        }

    def run_multi_experiment_analysis(
        self,
        base_directory: str,
        experiment_pattern: str = "**/analyzing_result",
        max_experiments: Optional[int] = None
    ) -> Dict[str, Any]:
        """다중 실험 분석"""
        print(f"🌐 다중 실험 분석: {base_directory}")
        
        # 실험 디렉토리 찾기
        experiment_dirs = []
        for root, dirs, files in os.walk(base_directory):
            if 'analyzing_result' in dirs:
                analyzing_result_path = os.path.join(root, 'analyzing_result')
                checkpoint_analysis_file = os.path.join(analyzing_result_path, 'checkpoint_analysis_results.csv')
                
                if os.path.exists(checkpoint_analysis_file):
                    experiment_dirs.append({
                        'name': os.path.basename(root),
                        'path': analyzing_result_path,
                        'analysis_file': checkpoint_analysis_file
                    })
        
        if max_experiments:
            experiment_dirs = experiment_dirs[:max_experiments]
        
        print(f"📊 발견된 실험: {len(experiment_dirs)}개")
        
        # 각 실험 분석
        experiment_results = []
        for exp_info in experiment_dirs:
            print(f"\n🎯 실험 처리: {exp_info['name']}")
            
            try:
                # 실험 설정 생성
                experiment_config = ExperimentConfig(
                    experiment_name=exp_info['name'],
                    analysis_result_path=exp_info['analysis_file'],
                    test_config_base_path=self._infer_test_config_path(exp_info['name']),
                    output_dir=os.path.join(exp_info['path'], 'pipeline_results')
                )
                
                # 실험 분석 실행
                result = self.run_experiment_analysis(experiment_config)
                experiment_results.append(result)
                
            except Exception as e:
                print(f"❌ 실험 분석 실패: {e}")
                experiment_results.append({
                    'experiment_name': exp_info['name'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'total_experiments': len(experiment_results),
            'successful_experiments': sum(1 for r in experiment_results if r.get('success_count', 0) > 0),
            'results': experiment_results
        }
    
    def _extract_model_info_from_checkpoint(self, checkpoint_path: str) -> Dict:
        """체크포인트에서 모델 정보 추출"""
        yaml_files = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*.yml'))
        
        if not yaml_files:
            return {'model_name': 'EEGNetLNL', 'batch_size': self.extraction_config.batch_size}
        
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            
            search_space = config.get('search_space', {})
            return {
                'model_name': search_space.get('model_name', 'EEGNetLNL'),
                'batch_size': search_space.get('batch_size', self.extraction_config.batch_size)
            }
        except:
            return {'model_name': 'EEGNetLNL', 'batch_size': self.extraction_config.batch_size}
    
    def _infer_test_config_path(self, experiment_name: str) -> str:
        """실험명에서 테스트 설정 경로 추론"""
        # 실험명 패턴에 따른 경로 매핑
        if 'wire' in experiment_name.lower():
            return os.path.join(self.extraction_config.test_data_path, "data/raw3config/test/only1Day1,8")
        elif 'wireless' in experiment_name.lower():
            return os.path.join(self.extraction_config.test_data_path, "data/raw5&6config/test/only1Day1,8")
        elif 'ui' in experiment_name.lower():
            return os.path.join(self.extraction_config.test_data_path, "src/config/data_config/UIconfig/test")
        elif 'unm' in experiment_name.lower():
            return os.path.join(self.extraction_config.test_data_path, "src/config/data_config/UNMconfig/test")
        else:
            # 기본값
            return os.path.join(self.extraction_config.test_data_path, "data/raw5&6config/test/only1Day1,8")
    
    def _combine_feature_data_list(self, feature_data_list: List) -> 'FeatureData':
        """다중 FeatureData를 하나로 결합"""
        from .feature_extractor import FeatureData
        import numpy as np
        
        if len(feature_data_list) == 1:
            return feature_data_list[0]
        
        # 모든 데이터 결합
        combined_features = np.vstack([fd.features for fd in feature_data_list])
        combined_target_labels = np.hstack([fd.target_labels for fd in feature_data_list])
        combined_input_data = np.vstack([fd.input_data for fd in feature_data_list])
        
        # Subject 라벨 생성
        combined_subject_labels = []
        subject_names = []
        
        for i, fd in enumerate(feature_data_list):
            subject_names.append(fd.subject_name)
            combined_subject_labels.extend([i] * len(fd.target_labels))
        
        # 도메인 라벨 결합 (있는 경우)
        combined_domain_labels = None
        if all(fd.domain_labels is not None for fd in feature_data_list):
            combined_domain_labels = np.hstack([fd.domain_labels for fd in feature_data_list])
        
        # 메타데이터 설정
        first_fd = feature_data_list[0]
        
        return FeatureData(
            features=combined_features,
            target_labels=combined_target_labels,
            input_data=combined_input_data,
            domain_labels=combined_domain_labels,
            subject_name="COMBINED",
            model_name=first_fd.model_name,
            checkpoint_path=first_fd.checkpoint_path,
            metadata={
                'subject_names': subject_names,
                'subject_labels': np.array(combined_subject_labels),
                'num_subjects': len(feature_data_list)
            }
        )