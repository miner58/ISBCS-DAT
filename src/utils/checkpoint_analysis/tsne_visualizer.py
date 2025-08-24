"""
t-SNE Visualizer Module

t-SNE 기반 시각화 및 비교 분석 모듈
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .feature_extractor import FeatureData


@dataclass
class VisualizationConfig:
    """시각화 설정"""
    output_formats: List[str] = field(default_factory=lambda: ['png', 'svg'])
    figure_size: Tuple[int, int] = (12, 6)
    dpi: int = 300
    style: str = "white"
    context: str = "notebook"
    colors: str = "Set3"
    alpha: float = 0.7
    marker_size: int = 50
    edge_width: float = 0.3
    tsne_params: Dict = field(default_factory=lambda: {
        'n_components': 2,
        'random_state': 42,
        'max_iter': 1000,
        'init': 'pca',
        'learning_rate': 'auto'
    })


class PlotStrategy(ABC):
    """시각화 전략 추상 클래스"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._setup_style()
    
    def _setup_style(self):
        """스타일 설정"""
        sns.set_theme(style=self.config.style, context=self.config.context)
    
    @abstractmethod
    def create_plot(self, *args, **kwargs) -> plt.Figure:
        """플롯 생성"""
        pass
    
    def save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """플롯 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        for fmt in self.config.output_formats:
            filepath = os.path.join(save_path, f"{filename}.{fmt}")
            
            if fmt == 'svg':
                fig.savefig(filepath, format='svg', bbox_inches='tight')
            else:
                fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            
            print(f"💾 저장: {filepath}")
    
    def _prepare_data_for_tsne(self, data: np.ndarray) -> np.ndarray:
        """t-SNE용 데이터 전처리"""
        if len(data.shape) > 2:
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
        
        scaler = StandardScaler()
        return scaler.fit_transform(data_flat)
    
    def _compute_tsne(self, data: np.ndarray) -> np.ndarray:
        """t-SNE 계산"""
        perplexity = min(30, data.shape[0] - 1)
        tsne_params = self.config.tsne_params.copy()
        tsne_params['perplexity'] = perplexity
        
        tsne = TSNE(**tsne_params)
        return tsne.fit_transform(data)
    
    def _get_color_map(self, labels: np.ndarray) -> Dict:
        """레이블별 색상 매핑"""
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(self.config.colors)(np.linspace(0, 1, len(unique_labels)))
        return {label: colors[i] for i, label in enumerate(unique_labels)}


class InputVsFeaturePlot(PlotStrategy):
    """입력 vs 특징 비교 플롯"""
    
    def create_plot(self, feature_data: FeatureData) -> plt.Figure:
        """입력 대 특징 비교 플롯 생성"""
        input_data = feature_data.input_data
        features = feature_data.features
        target_labels = feature_data.target_labels
        
        # 데이터 전처리
        input_scaled = self._prepare_data_for_tsne(input_data)
        features_scaled = self._prepare_data_for_tsne(features)
        
        # t-SNE 계산
        input_tsne = self._compute_tsne(input_scaled)
        features_tsne = self._compute_tsne(features_scaled)
        
        # 플롯 생성
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        color_map = self._get_color_map(target_labels)
        unique_labels = np.unique(target_labels)
        
        # 입력 데이터 t-SNE
        ax1 = axes[0]
        for label in unique_labels:
            mask = target_labels == label
            ax1.scatter(
                input_tsne[mask, 0], input_tsne[mask, 1],
                c=[color_map[label]], label=f'Class {label}',
                alpha=self.config.alpha, s=self.config.marker_size,
                edgecolors='black', linewidth=self.config.edge_width
            )
        
        ax1.set_title(f'Input Data t-SNE\n{feature_data.subject_name}', fontweight='bold')
        ax1.legend(frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        sns.despine(ax=ax1, left=True, bottom=True)
        
        # 특징 데이터 t-SNE
        ax2 = axes[1]
        for label in unique_labels:
            mask = target_labels == label
            ax2.scatter(
                features_tsne[mask, 0], features_tsne[mask, 1],
                c=[color_map[label]], label=f'Class {label}',
                alpha=self.config.alpha, s=self.config.marker_size,
                edgecolors='black', linewidth=self.config.edge_width
            )
        
        ax2.set_title(f'Extracted Features t-SNE\n{feature_data.model_name}', fontweight='bold')
        ax2.legend(frameon=False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        sns.despine(ax=ax2, left=True, bottom=True)
        
        plt.tight_layout()
        return fig


class CombinedSubjectsPlot(PlotStrategy):
    """다중 피험자 통합 플롯"""
    
    def create_plot(self, feature_data_list: List[FeatureData], experiment_name: str = "") -> plt.Figure:
        """다중 피험자 통합 플롯 생성"""
        # 데이터 결합
        all_input_data = []
        all_features_data = []
        all_target_labels = []
        all_subject_labels = []
        subject_names = []
        
        for i, feature_data in enumerate(feature_data_list):
            all_input_data.append(feature_data.input_data)
            all_features_data.append(feature_data.features)
            all_target_labels.append(feature_data.target_labels)
            all_subject_labels.extend([i] * len(feature_data.target_labels))
            subject_names.append(feature_data.subject_name)
        
        combined_input = np.vstack(all_input_data)
        combined_features = np.vstack(all_features_data)
        combined_target_labels = np.hstack(all_target_labels)
        combined_subject_labels = np.array(all_subject_labels)
        
        # 데이터 전처리 및 t-SNE
        input_scaled = self._prepare_data_for_tsne(combined_input)
        features_scaled = self._prepare_data_for_tsne(combined_features)
        
        input_tsne = self._compute_tsne(input_scaled)
        features_tsne = self._compute_tsne(features_scaled)
        
        # 플롯 생성
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        unique_labels = np.unique(combined_target_labels)
        unique_subjects = np.unique(combined_subject_labels)
        
        color_map = self._get_color_map(combined_target_labels)
        
        # 마커 매핑
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X', 'h', '+']
        marker_map = {subj: markers[i % len(markers)] for i, subj in enumerate(unique_subjects)}
        
        # 입력 데이터 t-SNE
        ax1 = axes[0]
        for subj_idx in unique_subjects:
            subj_name = subject_names[subj_idx]
            for label in unique_labels:
                mask = (combined_subject_labels == subj_idx) & (combined_target_labels == label)
                if np.any(mask):
                    ax1.scatter(
                        input_tsne[mask, 0], input_tsne[mask, 1],
                        c=[color_map[label]], marker=marker_map[subj_idx],
                        label=f'{subj_name}_C{label}',
                        alpha=self.config.alpha, s=60,
                        edgecolors='black', linewidth=self.config.edge_width
                    )
        
        ax1.set_title(f'Input Data t-SNE - All Subjects\n{experiment_name}', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])
        sns.despine(ax=ax1, left=True, bottom=True)
        
        # 특징 데이터 t-SNE
        ax2 = axes[1]
        for subj_idx in unique_subjects:
            subj_name = subject_names[subj_idx]
            for label in unique_labels:
                mask = (combined_subject_labels == subj_idx) & (combined_target_labels == label)
                if np.any(mask):
                    ax2.scatter(
                        features_tsne[mask, 0], features_tsne[mask, 1],
                        c=[color_map[label]], marker=marker_map[subj_idx],
                        label=f'{subj_name}_C{label}',
                        alpha=self.config.alpha, s=60,
                        edgecolors='black', linewidth=self.config.edge_width
                    )
        
        model_name = feature_data_list[0].model_name if feature_data_list else "Model"
        ax2.set_title(f'Extracted Features t-SNE - All Subjects\n{model_name}', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        sns.despine(ax=ax2, left=True, bottom=True)
        
        plt.tight_layout()
        return fig


class AugmentationComparisonPlot(PlotStrategy):
    """데이터 증강 비교 플롯"""
    
    def create_plot(
        self,
        original_data: Union[FeatureData, List[FeatureData]],
        augmented_data: Union[FeatureData, List[FeatureData]],
        plot_type: str = "type1",
        color_by: str = "subject",
        subject_name: str = "ALL"
    ) -> plt.Figure:
        """데이터 증강 비교 플롯 생성
        
        Args:
            original_data: 원본 특징 데이터 (단일 또는 리스트)
            augmented_data: 증강 특징 데이터 (단일 또는 리스트)
            plot_type: 플롯 타입 ("type1", "type2", "type3")
            color_by: 색상 구분 기준 ("class" 또는 "subject")
            subject_name: 피험자명 (단일 피험자명 또는 "ALL")
        """

        # 입력 검증
        if color_by not in ["class", "subject"]:
            raise ValueError("color_by는 'class' 또는 'subject'여야 합니다")
        
        # # MULTI 모드 또는 subject 색상일 때 List 입력 검증
        # if (subject_name == "MULTI" or (color_by == "subject" and subject_name == "ALL")):
        #     if not isinstance(original_data, list) or not isinstance(augmented_data, list):
        #         if subject_name == "MULTI":
        #             raise ValueError("MULTI 모드에서는 List[FeatureData] 입력이 필요합니다")
        #         else:
        #             raise ValueError("ALL 모드에서는 List[FeatureData] 입력이 필요합니다")
            
        #     if len(original_data) != len(augmented_data):
        #         raise ValueError("원본과 증강 데이터의 피험자 수가 일치하지 않습니다")
        
        # 데이터 정규화 (단일 -> 리스트 형태로 통일)
        if isinstance(original_data, FeatureData):
            original_list = [original_data]
            augmented_list = [augmented_data]
            subject_names = [original_data.subject_name or "Subject1"]
        else:
            original_list = original_data
            augmented_list = augmented_data
            subject_names = [data.subject_name or f"Subject{i+1}" for i, data in enumerate(original_data)]
        
        # 플롯 데이터 구성
        if plot_type == "type1":
            # 원본 입력 + 증강된 입력 + 증강된 특징
            plot_configs = [
                ("Original Input", "input_data", original_list),
                ("Augmented Input", "input_data", augmented_list),
                ("Augmented Features", "features", augmented_list)
            ]
            title_suffix = "Original vs Augmented Input + Aug Features"
            
        elif plot_type == "type2":
            # 원본 입력 + 원본 특징
            plot_configs = [
                ("Original Input", "input_data", original_list),
                ("Original Features", "features", original_list)
            ]
            title_suffix = "Original Input vs Features"
            
        elif plot_type == "type3":
            # 증강된 입력 + 증강된 특징
            plot_configs = [
                ("Augmented Input", "input_data", augmented_list),
                ("Augmented Features", "features", augmented_list)
            ]
            title_suffix = "Augmented Input vs Features"
            
        else:
            raise ValueError(f"지원하지 않는 plot_type: {plot_type}")
        
        # 제목에 피험자 정보 추가
        if subject_name == "MULTI":
            title_suffix = f"{title_suffix} - Multi Subjects"
        elif subject_name != "ALL":
            title_suffix = f"{title_suffix} - {subject_name}"
        else:
            title_suffix = f"{title_suffix} - All Subjects"
        
        n_plots = len(plot_configs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        # 클래스별 마커 정의
        class_markers = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'v', 5: '<', 6: '>', 7: 'P'}
        
        for idx, (name, data_type, data_list) in enumerate(plot_configs):
            ax = axes[idx]
            
            # 데이터 결합
            combined_data = []
            combined_target_labels = []
            combined_subject_labels = []
            combined_domain_labels = []  # 🆕 domain_labels 추가
            
            for subj_idx, data in enumerate(data_list):
                if data_type == "input_data":
                    combined_data.append(data.input_data)
                else:  # "features"
                    combined_data.append(data.features)
                
                combined_target_labels.append(data.target_labels)
                combined_subject_labels.extend([subj_idx] * len(data.target_labels))
                
                # 🆕 domain_labels 수집
                if hasattr(data, 'domain_labels') and data.domain_labels is not None:
                    combined_domain_labels.extend(data.domain_labels)
                else:
                    # fallback: subject index 사용
                    combined_domain_labels.extend([subj_idx] * len(data.target_labels))
            
            combined_data_array = np.vstack(combined_data)
            combined_target_labels_array = np.hstack(combined_target_labels)
            combined_subject_labels_array = np.array(combined_subject_labels)
            combined_domain_labels_array = np.array(combined_domain_labels)  # 🆕 domain_labels 배열 생성
            
            # t-SNE 계산
            data_scaled = self._prepare_data_for_tsne(combined_data_array)
            data_tsne = self._compute_tsne(data_scaled)
            
            unique_target_labels = np.unique(combined_target_labels_array)
            unique_subject_indices = np.unique(combined_subject_labels_array)
            
            if color_by == "class":
                print(f"color_by: {color_by}, subject_name: {subject_name}")
                # 기존 방식: 클래스별 색상
                color_map = self._get_color_map(combined_target_labels_array)
                
                for label in unique_target_labels:
                    mask = combined_target_labels_array == label
                    ax.scatter(
                        data_tsne[mask, 0], data_tsne[mask, 1],
                        c=[color_map[label]], label=f'Class {label}',
                        alpha=self.config.alpha, s=self.config.marker_size,
                        edgecolors='black', linewidth=self.config.edge_width
                    )
                
            elif color_by == "subject":
                print(f"color_by: {color_by}, subject_name: {subject_name}")
                # 🆕 subject_name="ALL"일 때 domain_labels 기준으로 색상 구분
                if subject_name == "ALL":
                    print("ALL subject_name, using domain_labels for color distinction")
                    # domain_labels 기준 색상 구분
                    color_labels = combined_domain_labels_array
                    unique_color_labels = np.unique(color_labels)
                    color_map = self._get_color_map(color_labels)
                    label_prefix = "Domain"
                    
                    for color_label in unique_color_labels:
                        for class_label in unique_target_labels:
                            mask = (color_labels == color_label) & (combined_target_labels_array == class_label)
                            if np.any(mask):
                                marker = class_markers.get(class_label, 'o')
                                ax.scatter(
                                    data_tsne[mask, 0], data_tsne[mask, 1],
                                    c=[color_map[color_label]], 
                                    # marker=marker,
                                    label=f'{label_prefix} {color_label}, Class {class_label}',
                                    alpha=self.config.alpha,
                                    s=self.config.marker_size,
                                    edgecolors='black',
                                    linewidth=self.config.edge_width
                                )
                else:
                    print(f"Single subject: {subject_name}, using subject labels for color distinction")
                    # 기존 방식: 피험자별 색상 + 클래스별 마커
                    subject_color_map = self._get_color_map(combined_subject_labels_array)
                    
                    for subj_idx in unique_subject_indices:
                        subj_name = subject_names[subj_idx]
                        subj_color = subject_color_map[subj_idx]
                        
                        for class_label in unique_target_labels:
                            mask = (combined_subject_labels_array == subj_idx) & (combined_target_labels_array == class_label)
                            if np.any(mask):
                                marker = class_markers.get(class_label, 'o')
                                ax.scatter(
                                    data_tsne[mask, 0], data_tsne[mask, 1],
                                    c=[subj_color], 
                                    # marker=marker,
                                    label=f'{subj_name}_C{class_label}',
                                    alpha=self.config.alpha,
                                    s=self.config.marker_size,
                                    edgecolors='black', 
                                linewidth=self.config.edge_width
                            )
            
            ax.set_title(f'{name} t-SNE', fontweight='bold')
            ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)
        
        plt.suptitle(title_suffix, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


class TSNEVisualizer:
    """t-SNE 시각화 메인 클래스"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self._strategies = {
            'input_vs_feature': InputVsFeaturePlot(self.config),
            'combined_subjects': CombinedSubjectsPlot(self.config),
            'augmentation_comparison': AugmentationComparisonPlot(self.config)
        }
    
    def create_input_vs_feature_plot(
        self,
        feature_data: FeatureData,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """입력 vs 특징 비교 플롯"""
        strategy = self._strategies['input_vs_feature']
        fig = strategy.create_plot(feature_data)
        
        if save_path:
            filename = f"{feature_data.subject_name}_input_vs_features_tsne"
            strategy.save_plot(fig, save_path, filename)
        
        return fig
    
    def create_combined_subjects_plot(
        self,
        feature_data_list: List[FeatureData],
        experiment_name: str = "",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """다중 피험자 통합 플롯"""
        strategy = self._strategies['combined_subjects']
        fig = strategy.create_plot(feature_data_list, experiment_name)
        
        if save_path:
            filename = f"{experiment_name}_all_subjects_combined_tsne"
            strategy.save_plot(fig, save_path, filename)
        
        return fig
    
    def create_augmentation_comparison_plot(
        self,
        original_data: FeatureData,
        augmented_data: FeatureData,
        plot_type: str = "type1",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """데이터 증강 비교 플롯"""
        strategy = self._strategies['augmentation_comparison']
        fig = strategy.create_plot(original_data, augmented_data, plot_type)
        
        if save_path:
            filename = f"augmentation_comparison_{plot_type}"
            strategy.save_plot(fig, save_path, filename)
        
        return fig
    
    def load_feature_data_from_npz(self, npz_path: str) -> FeatureData:
        """NPZ 파일에서 특징 데이터 로드"""
        data = np.load(npz_path)
        
        return FeatureData(
            features=data['features'],
            target_labels=data['target_labels'],
            input_data=data['input_data'],
            domain_labels=data.get('domain_labels', None),
            subject_name=str(data.get('subject_name', '')),
            model_name=str(data.get('model_name', '')),
            checkpoint_path=str(data.get('checkpoint_path', ''))
        )