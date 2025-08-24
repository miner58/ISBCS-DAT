"""
Checkpoint Analysis Package

체크포인트 기반 특징 추출 및 t-SNE 시각화를 위한 통합 패키지
"""

from .feature_extractor import (
    FeatureExtractor,
    ExtractionConfig,
    FeatureData,
    ModelRegistry,
    DataModuleFactory
)

from .tsne_visualizer import (
    TSNEVisualizer,
    VisualizationConfig,
    PlotStrategy,
    InputVsFeaturePlot,
    CombinedSubjectsPlot,
    AugmentationComparisonPlot
)

from .analysis_pipeline import (
    AnalysisPipeline,
    PipelineConfig,
    SubjectConfig,
    ExperimentConfig
)

__all__ = [
    # Feature Extraction
    'FeatureExtractor',
    'ExtractionConfig', 
    'FeatureData',
    'ModelRegistry',
    'DataModuleFactory',
    
    # Visualization
    'TSNEVisualizer',
    'VisualizationConfig',
    'PlotStrategy',
    'InputVsFeaturePlot',
    'CombinedSubjectsPlot', 
    'AugmentationComparisonPlot',
    
    # Pipeline
    'AnalysisPipeline',
    'PipelineConfig',
    'SubjectConfig',
    'ExperimentConfig'
]

__version__ = "1.0.0"
__author__ = "EEG-BCI Fairness Research Team"