"""
Input vs Feature t-SNE 비교 시각화 모듈

추출된 NPZ 파일에서 input_data와 features를 로드하여 t-SNE 비교 시각화
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import glob
import warnings
warnings.filterwarnings('ignore')


def load_extracted_features(npz_file_path: str) -> Dict:
    """NPZ 파일에서 데이터 로드"""
    data = np.load(npz_file_path)
    return {
        'input_data': data['input_data'],
        'features': data['features'], 
        'target_labels': data['target_labels'],
        'domain_labels': data.get('domain_labels', None),
        'subject_name': str(data['subject_name']),
        'model_name': str(data['model_name'])
    }


def load_extracted_features_with_predictions(npz_file_path: str) -> Dict:
    """예측 정보를 포함한 NPZ 파일에서 데이터 로드"""
    data = np.load(npz_file_path)
    result = {
        'input_data': data['input_data'],
        'features': data['features'], 
        'target_labels': data['target_labels'],
        'subject_name': str(data['subject_name']),
        'model_name': str(data['model_name'])
    }
    
    # 예측 정보가 있는 경우 추가
    if 'predictions' in data:
        result.update({
            'predictions': data['predictions'],
            'prediction_probs': data['prediction_probs'],
            'correct_predictions': data['correct_predictions'],
            'accuracy': float(data['accuracy']),
            'per_class_accuracy': eval(str(data['per_class_accuracy'])) if 'per_class_accuracy' in data else {}
        })
    
    # 도메인 라벨이 있는 경우 추가
    if 'domain_labels' in data and data['domain_labels'] is not None:
        result['domain_labels'] = data['domain_labels']
    
    return result


def plot_prediction_accuracy_tsne(data_dict: Dict, save_path: Optional[str] = None):
    """예측 정확도에 따른 색상 구분 t-SNE (Type A)"""
    features = data_dict['features']
    target_labels = data_dict['target_labels'] 
    correct_predictions = data_dict['correct_predictions']
    subject_name = data_dict['subject_name']
    model_name = data_dict['model_name']
    accuracy = data_dict['accuracy']
    
    print(f"🎯 {subject_name} 예측 정확도 t-SNE 시작 (정확도: {accuracy:.4f})")
    
    # 데이터 전처리
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE 수행
    perplexity_val = min(30, features.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, 
               max_iter=1000, init='pca', learning_rate='auto')
    features_tsne = tsne.fit_transform(features_scaled)
    
    # 시각화
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 클래스별 색상
    unique_labels = np.unique(target_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 정확한 예측: 진한 색상 + 원형 마커, 틀린 예측: 연한 색상 + X 마커
    for class_id in unique_labels:
        class_mask = target_labels == class_id
        
        # 정확한 예측
        correct_mask = class_mask & correct_predictions
        if np.any(correct_mask):
            ax.scatter(features_tsne[correct_mask, 0], features_tsne[correct_mask, 1],
                      c=[color_map[class_id]], marker='o', s=60, alpha=0.8,
                      label=f'Class {class_id} (Correct)', edgecolors='black', linewidth=0.5)
        
        # 틀린 예측  
        incorrect_mask = class_mask & ~correct_predictions
        if np.any(incorrect_mask):
            ax.scatter(features_tsne[incorrect_mask, 0], features_tsne[incorrect_mask, 1],
                      c=[color_map[class_id]], marker='x', s=100, alpha=0.9,
                      label=f'Class {class_id} (Wrong)', linewidths=3)
    
    ax.set_title(f'Prediction Accuracy Analysis\n{subject_name} ({model_name}) - Accuracy: {accuracy:.4f}', 
                fontweight='bold', fontsize=12)
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        base_filename = f"{subject_name}_prediction_accuracy_tsne"
        
        # PNG 저장
        png_filename = f"{base_filename}.png"
        plt.savefig(os.path.join(save_path, png_filename), dpi=300, bbox_inches='tight')
        print(f"💾 PNG 저장: {png_filename}")
        
        # SVG 저장
        svg_filename = f"{base_filename}.svg"
        plt.savefig(os.path.join(save_path, svg_filename), format='svg', bbox_inches='tight')
        print(f"💾 SVG 저장: {svg_filename}")
    
    plt.show()


def plot_prediction_confidence_tsne(data_dict: Dict, save_path: Optional[str] = None):
    """예측 신뢰도에 따른 크기 구분 t-SNE (Type B)"""
    features = data_dict['features']
    target_labels = data_dict['target_labels']
    prediction_probs = data_dict['prediction_probs']
    correct_predictions = data_dict['correct_predictions']
    subject_name = data_dict['subject_name']
    model_name = data_dict['model_name']
    accuracy = data_dict['accuracy']
    
    print(f"🎯 {subject_name} 예측 신뢰도 t-SNE 시작")
    
    # 예측 신뢰도 계산 (최대 확률값)
    confidence = np.max(prediction_probs, axis=1)
    
    # 데이터 전처리
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE 수행
    perplexity_val = min(30, features.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
               max_iter=1000, init='pca', learning_rate='auto')
    features_tsne = tsne.fit_transform(features_scaled)
    
    # 시각화
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 클래스별 색상
    unique_labels = np.unique(target_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for class_id in unique_labels:
        class_mask = target_labels == class_id
        
        # 정확한 예측 (원형 마커, 신뢰도에 비례하는 크기)
        correct_mask = class_mask & correct_predictions
        if np.any(correct_mask):
            ax.scatter(features_tsne[correct_mask, 0], features_tsne[correct_mask, 1],
                      c=[color_map[class_id]], s=confidence[correct_mask] * 200,  # 신뢰도에 비례하는 크기
                      alpha=0.7, marker='o', edgecolors='black', linewidth=0.5,
                      label=f'Class {class_id} (Correct)')
        
        # 틀린 예측 (사각형 마커, 빨간 테두리)
        incorrect_mask = class_mask & ~correct_predictions  
        if np.any(incorrect_mask):
            ax.scatter(features_tsne[incorrect_mask, 0], features_tsne[incorrect_mask, 1],
                      c=[color_map[class_id]], s=confidence[incorrect_mask] * 200,
                      alpha=0.7, marker='s', edgecolors='red', linewidths=2,
                      label=f'Class {class_id} (Wrong)')
    
    ax.set_title(f'Prediction Confidence Analysis\n{subject_name} ({model_name}) - Accuracy: {accuracy:.4f}', 
                fontweight='bold', fontsize=12)
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    
    # 신뢰도 범례 추가
    confidence_legend = [plt.scatter([], [], s=conf*200, c='gray', alpha=0.7, edgecolors='black') 
                        for conf in [0.5, 0.7, 0.9]]
    legend2 = ax.legend(confidence_legend, ['Low Conf (0.5)', 'Med Conf (0.7)', 'High Conf (0.9)'], 
                       loc='upper right', title='Confidence', frameon=False)
    ax.add_artist(ax.legend_)  # 클래스 범례 유지
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        base_filename = f"{subject_name}_prediction_confidence_tsne"
        
        # PNG 저장
        png_filename = f"{base_filename}.png"
        plt.savefig(os.path.join(save_path, png_filename), dpi=300, bbox_inches='tight')
        print(f"💾 PNG 저장: {png_filename}")
        
        # SVG 저장
        svg_filename = f"{base_filename}.svg"
        plt.savefig(os.path.join(save_path, svg_filename), format='svg', bbox_inches='tight')
        print(f"💾 SVG 저장: {svg_filename}")
    
    plt.show()


def analyze_misclassification_patterns(data_dict: Dict, save_path: Optional[str] = None):
    """오분류 패턴 상세 분석"""
    features = data_dict['features']
    target_labels = data_dict['target_labels']
    predictions = data_dict['predictions']
    correct_predictions = data_dict['correct_predictions']
    subject_name = data_dict['subject_name']
    model_name = data_dict['model_name']
    accuracy = data_dict['accuracy']
    
    print(f"🎯 {subject_name} 오분류 패턴 분석 시작")
    
    # 오분류 샘플들만 추출
    misclassified_mask = ~correct_predictions
    
    if not np.any(misclassified_mask):
        print(f"✅ {subject_name}: 오분류 샘플이 없습니다 (완벽한 성능!)")
        return
    
    misclassified_features = features[misclassified_mask]
    misclassified_true = target_labels[misclassified_mask]
    misclassified_pred = predictions[misclassified_mask]
    
    print(f"   오분류 샘플 수: {len(misclassified_features)} / {len(features)}")
    
    # 오분류 유형별 분석
    confusion_types = {}
    for true_class in np.unique(target_labels):
        for pred_class in np.unique(predictions):
            if true_class != pred_class:
                mask = (misclassified_true == true_class) & (misclassified_pred == pred_class)
                if np.any(mask):
                    confusion_types[f'{true_class}→{pred_class}'] = np.sum(mask)
    
    if not confusion_types:
        print(f"   혼동 패턴이 발견되지 않았습니다.")
        return
    
    print(f"   발견된 혼동 패턴: {confusion_types}")
    
    # 전체 특징과 오분류 특징 함께 시각화
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE 수행
    perplexity_val = min(30, features.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
               max_iter=1000, init='pca', learning_rate='auto')
    features_tsne = tsne.fit_transform(features_scaled)
    
    # 시각화
    sns.set_theme(style="white", context="notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 클래스별 색상
    unique_labels = np.unique(target_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 좌측: 전체 데이터 (정확/부정확 구분)
    for class_id in unique_labels:
        class_mask = target_labels == class_id
        
        # 정확한 예측
        correct_mask = class_mask & correct_predictions
        if np.any(correct_mask):
            ax1.scatter(features_tsne[correct_mask, 0], features_tsne[correct_mask, 1],
                       c=[color_map[class_id]], marker='o', s=40, alpha=0.6,
                       label=f'Class {class_id} (Correct)', edgecolors='black', linewidth=0.3)
        
        # 틀린 예측 (강조)
        incorrect_mask = class_mask & ~correct_predictions
        if np.any(incorrect_mask):
            ax1.scatter(features_tsne[incorrect_mask, 0], features_tsne[incorrect_mask, 1],
                       c=[color_map[class_id]], marker='X', s=100, alpha=1.0,
                       label=f'Class {class_id} (Wrong)', edgecolors='red', linewidths=2)
    
    ax1.set_title(f'All Samples with Misclassifications Highlighted\n{subject_name} (Acc: {accuracy:.4f})', 
                 fontweight='bold')
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(ax=ax1, left=True, bottom=True)
    
    # 우측: 오분류 샘플만 (혼동 유형별)
    misclassified_tsne = features_tsne[misclassified_mask]
    confusion_colors = plt.cm.Set1(np.linspace(0, 1, len(confusion_types)))
    
    for i, (conf_type, count) in enumerate(confusion_types.items()):
        true_class, pred_class = map(int, conf_type.split('→'))
        type_mask = (misclassified_true == true_class) & (misclassified_pred == pred_class)
        
        if np.any(type_mask):
            ax2.scatter(misclassified_tsne[type_mask, 0], misclassified_tsne[type_mask, 1],
                       c=[confusion_colors[i]], s=80, alpha=0.8,
                       label=f'{conf_type} ({count} samples)', 
                       edgecolors='black', linewidth=0.5)
    
    ax2.set_title(f'Misclassification Patterns in Feature Space\n{model_name}', fontweight='bold')
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, bottom=True)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        base_filename = f"{subject_name}_misclassification_analysis"
        
        # PNG 저장
        png_filename = f"{base_filename}.png"
        plt.savefig(os.path.join(save_path, png_filename), dpi=300, bbox_inches='tight')
        print(f"💾 PNG 저장: {png_filename}")
        
        # SVG 저장
        svg_filename = f"{base_filename}.svg"
        plt.savefig(os.path.join(save_path, svg_filename), format='svg', bbox_inches='tight')
        print(f"💾 SVG 저장: {svg_filename}")
    
    plt.show()


def calculate_feature_space_metrics(data_dict: Dict) -> Dict:
    """특징 공간에서의 정량적 메트릭 계산"""
    features = data_dict['features']
    target_labels = data_dict['target_labels']
    predictions = data_dict['predictions']
    correct_predictions = data_dict['correct_predictions']
    
    metrics = {}
    
    # 1. 클래스별 정확도
    for class_id in np.unique(target_labels):
        class_mask = target_labels == class_id
        if np.sum(class_mask) > 0:
            metrics[f'class_{class_id}_accuracy'] = np.mean(correct_predictions[class_mask])
    
    # 2. 특징 공간에서의 클래스 분리도 (실루엣 스코어)
    from sklearn.metrics import silhouette_score
    if len(np.unique(target_labels)) > 1:
        try:
            metrics['silhouette_score_true'] = silhouette_score(features, target_labels)
            metrics['silhouette_score_pred'] = silhouette_score(features, predictions)
        except:
            metrics['silhouette_score_true'] = 0.0
            metrics['silhouette_score_pred'] = 0.0
    
    # 3. 오분류 영역의 특징 공간 밀도
    misclassified_features = features[~correct_predictions]
    if len(misclassified_features) > 1:
        try:
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(misclassified_features)
            metrics['misclassification_density'] = np.mean(distances[np.triu_indices_from(distances, k=1)])
        except:
            metrics['misclassification_density'] = 0.0
    
    # 4. 전체 정확도 및 기본 통계
    metrics['overall_accuracy'] = np.mean(correct_predictions)
    metrics['num_samples'] = len(features)
    metrics['num_misclassified'] = np.sum(~correct_predictions)
    
    return metrics


def plot_prediction_analysis_tsne(npz_file_path: str, save_path: Optional[str] = None, 
                                 analysis_type: str = 'accuracy'):
    """예측 분석 기반 t-SNE 시각화
    
    Args:
        npz_file_path: NPZ 파일 경로
        save_path: 저장 경로
        analysis_type: 'accuracy', 'confidence', 'boundary', 'misclassification'
    """
    # 예측 정보를 포함한 데이터 로드
    data_dict = load_extracted_features_with_predictions(npz_file_path)
    
    # 예측 정보가 없는 경우 에러
    if 'predictions' not in data_dict:
        print(f"❌ {npz_file_path}: 예측 정보가 없습니다. 예측 포함 모드로 특징을 추출해주세요.")
        return False
    
    print(f"🎯 {data_dict['subject_name']} - {analysis_type} 분석 시작")
    
    try:
        if analysis_type == 'accuracy':
            plot_prediction_accuracy_tsne(data_dict, save_path)
        elif analysis_type == 'confidence':
            plot_prediction_confidence_tsne(data_dict, save_path)
        elif analysis_type == 'misclassification':
            analyze_misclassification_patterns(data_dict, save_path)
        else:
            print(f"❌ 지원하지 않는 분석 타입: {analysis_type}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ {analysis_type} 분석 실패: {str(e)}")
        return False


def prepare_data_for_tsne(input_data: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """t-SNE를 위한 데이터 전처리"""
    # 입력 데이터 평면화 (samples, channels*timepoints)
    input_flat = input_data.reshape(input_data.shape[0], -1)
    
    # 표준화
    scaler_input = StandardScaler()
    scaler_features = StandardScaler()
    
    input_scaled = scaler_input.fit_transform(input_flat)
    features_scaled = scaler_features.fit_transform(features)
    
    return input_scaled, features_scaled


def plot_input_vs_feature_tsne(data_dict: Dict, save_path: Optional[str] = None):
    """입력 vs 특징 t-SNE 비교 플롯"""
    input_data = data_dict['input_data']
    features = data_dict['features']
    target_labels = data_dict['target_labels']
    subject_name = data_dict['subject_name']
    model_name = data_dict['model_name']
    
    print(f"🎯 {subject_name} ({model_name}) t-SNE 비교 시작")
    print(f"   Input: {input_data.shape}, Features: {features.shape}")
    
    # 데이터 전처리
    input_scaled, features_scaled = prepare_data_for_tsne(input_data, features)
    
    # t-SNE 실행
    perplexity_val = min(30, input_data.shape[0] - 1)
    
    tsne_input = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, 
                     max_iter=1000, init='pca', learning_rate='auto')
    tsne_features = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
                        max_iter=1000, init='pca', learning_rate='auto')
    
    input_tsne = tsne_input.fit_transform(input_scaled)
    features_tsne = tsne_features.fit_transform(features_scaled)
    
    # 플롯 생성
    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 클래스별 색상
    unique_labels = np.unique(target_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 입력 데이터 t-SNE
    ax1 = axes[0]
    for label in unique_labels:
        mask = target_labels == label
        ax1.scatter(input_tsne[mask, 0], input_tsne[mask, 1], 
                   c=[color_map[label]], label=f'Class {label}',
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
    ax1.set_title(f'Input Data t-SNE\n{subject_name}', fontweight='bold')
    ax1.legend(frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(ax=ax1, left=True, bottom=True)
    
    # 특징 데이터 t-SNE  
    ax2 = axes[1]
    for label in unique_labels:
        mask = target_labels == label
        ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   c=[color_map[label]], label=f'Class {label}', 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
    ax2.set_title(f'Extracted Features t-SNE\n{model_name}', fontweight='bold')
    ax2.legend(frameon=False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, bottom=True)
    
    plt.tight_layout()
    
    # 저장 (PNG와 SVG 둘 다)
    if save_path:
        base_filename = f"{subject_name}_input_vs_features_tsne"
        
        # PNG 저장
        png_filename = f"{base_filename}.png"
        plt.savefig(os.path.join(save_path, png_filename), dpi=300, bbox_inches='tight')
        print(f"💾 PNG 저장: {png_filename}")
        
        # SVG 저장
        svg_filename = f"{base_filename}.svg"
        plt.savefig(os.path.join(save_path, svg_filename), format='svg', bbox_inches='tight')
        print(f"💾 SVG 저장: {svg_filename}")
    
    plt.show()


def plot_all_subjects_combined_tsne(experiment_dir: str, save_path: Optional[str] = None):
    """실험별로 모든 개체를 하나의 그래프에 표시"""
    # 해당 실험의 NPZ 파일들 찾기 (recursive search)
    npz_pattern = os.path.join(experiment_dir, "extracted_features", "**","*_features.npz")
    npz_files = glob.glob(npz_pattern, recursive=True)
    
    if not npz_files:
        print(f"❌ NPZ 파일이 없습니다: {experiment_dir}")
        print(f"   검색 패턴: {npz_pattern}")
        return
    
    print(f"🔍 실험 {os.path.basename(experiment_dir)}: {len(npz_files)}개 개체")
    
    # 모든 데이터 수집
    all_input_data = []
    all_features_data = []
    all_target_labels = []
    all_subject_labels = []
    
    subject_names = []
    
    for npz_file in npz_files:
        data_dict = load_extracted_features(npz_file)
        subject_name = data_dict['subject_name']
        subject_names.append(subject_name)
        
        # 데이터 추가
        all_input_data.append(data_dict['input_data'])
        all_features_data.append(data_dict['features'])
        all_target_labels.append(data_dict['target_labels'])
        all_subject_labels.extend([len(subject_names)-1] * len(data_dict['target_labels']))
    
    # 데이터 결합
    combined_input = np.vstack(all_input_data)
    combined_features = np.vstack(all_features_data)
    combined_target_labels = np.hstack(all_target_labels)
    combined_subject_labels = np.array(all_subject_labels)
    
    print(f"📊 결합된 데이터: Input {combined_input.shape}, Features {combined_features.shape}")
    
    # 데이터 전처리
    input_scaled, features_scaled = prepare_data_for_tsne(combined_input, combined_features)
    
    # t-SNE 실행
    perplexity_val = min(30, combined_input.shape[0] - 1)
    
    tsne_input = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, 
                     max_iter=1000, init='pca', learning_rate='auto')
    tsne_features = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
                        max_iter=1000, init='pca', learning_rate='auto')
    
    input_tsne = tsne_input.fit_transform(input_scaled)
    features_tsne = tsne_features.fit_transform(features_scaled)
    
    # 플롯 생성
    sns.set_theme(style="white", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 클래스별 색상 및 개체별 마커
    unique_labels = np.unique(combined_target_labels)
    unique_subjects = np.unique(combined_subject_labels)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X', 'h', '+']
    marker_map = {subj: markers[i % len(markers)] for i, subj in enumerate(unique_subjects)}
    
    # 입력 데이터 t-SNE
    ax1 = axes[0]
    for subj_idx in unique_subjects:
        subj_name = subject_names[subj_idx]
        for label in unique_labels:
            mask = (combined_subject_labels == subj_idx) & (combined_target_labels == label)
            if np.any(mask):
                ax1.scatter(input_tsne[mask, 0], input_tsne[mask, 1], 
                           c=[color_map[label]], marker=marker_map[subj_idx],
                           label=f'{subj_name}_C{label}', alpha=0.7, s=60, 
                           edgecolors='black', linewidth=0.3)
    
    ax1.set_title(f'Input Data t-SNE - All Subjects\n{os.path.basename(experiment_dir)}', fontweight='bold')
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
                ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                           c=[color_map[label]], marker=marker_map[subj_idx],
                           label=f'{subj_name}_C{label}', alpha=0.7, s=60, 
                           edgecolors='black', linewidth=0.3)
    
    ax2.set_title(f'Extracted Features t-SNE - All Subjects\nEEGNetLNL', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, bottom=True)
    
    plt.tight_layout()
    
    # 저장 경로 설정
    if save_path is None:
        save_path = experiment_dir
    
    # 저장 (PNG와 SVG 둘 다)
    base_filename = f"{os.path.basename(experiment_dir)}_all_subjects_combined_tsne"
    
    # PNG 저장
    png_filename = f"{base_filename}.png"
    plt.savefig(os.path.join(save_path, png_filename), dpi=300, bbox_inches='tight')
    print(f"💾 PNG 저장: {os.path.join(save_path, png_filename)}")
    
    # SVG 저장
    svg_filename = f"{base_filename}.svg"
    plt.savefig(os.path.join(save_path, svg_filename), format='svg', bbox_inches='tight')
    print(f"💾 SVG 저장: {os.path.join(save_path, svg_filename)}")
    
    plt.show()


def analyze_all_extracted_features(base_dir: str, save_path: Optional[str] = None):
    """모든 추출된 특징에 대해 input vs feature t-SNE 분석"""
    
    # NPZ 파일 찾기
    npz_pattern = os.path.join(base_dir, "**", "*_features.npz")
    npz_files = glob.glob(npz_pattern, recursive=True)
    
    if not npz_files:
        print(f"❌ NPZ 파일을 찾을 수 없습니다: {base_dir}")
        return
    
    print(f"🔍 발견된 NPZ 파일: {len(npz_files)}개")
    
    for npz_file in npz_files:
        print(f"\n📊 처리 중: {os.path.basename(npz_file)}")
        
        try:
            # 데이터 로드
            data_dict = load_extracted_features(npz_file)
            
            # t-SNE 비교 플롯 생성
            plot_input_vs_feature_tsne(data_dict, save_path)
            
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            continue
    
    print(f"\n✅ 분석 완료!")


def compare_input_feature_single(npz_file_path: str, save_path: Optional[str] = None):
    """단일 NPZ 파일에 대한 input vs feature 비교"""
    print(f"🎯 단일 파일 분석: {os.path.basename(npz_file_path)}")
    
    try:
        data_dict = load_extracted_features(npz_file_path)
        plot_input_vs_feature_tsne(data_dict, save_path)
        return True
    except Exception as e:
        print(f"❌ 분석 실패: {str(e)}")
        return False


def analyze_experiments_combined(base_dir: str, save_path: Optional[str] = None):
    """실험별로 모든 개체를 결합하여 분석"""
    
    # 실험 디렉토리들 찾기 (analyzing_result가 있는 폴더들)
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'analyzing_result' in dirs:
            analyzing_result_path = os.path.join(root, 'analyzing_result')
            if os.path.exists(os.path.join(analyzing_result_path, 'extracted_features')):
                experiment_dirs.append(analyzing_result_path)
    
    if not experiment_dirs:
        print(f"❌ 실험 디렉토리를 찾을 수 없습니다: {base_dir}")
        return
    
    # 경로 검증
    for exp_dir in experiment_dirs:
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"실험 디렉토리가 존재하지 않습니다: {exp_dir}")
        
    print(f"🔍 발견된 실험: {len(experiment_dirs)}개")
    
    for exp_dir in experiment_dirs:
        experiment_name = os.path.basename(os.path.dirname(exp_dir))
        print(f"\n📊 실험 처리 중: {experiment_name}")
        
        try:
            plot_all_subjects_combined_tsne(exp_dir, save_path)
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            continue
    
    print(f"\n✅ 모든 실험 분석 완료!")


def plot_augmentation_comparison_tsne(data_dict: Dict, save_path: Optional[str] = None, plot_type: str = "type1"):
    """데이터 증강 비교 t-SNE 플롯
    
    Args:
        data_dict: 원본과 증강된 데이터를 포함한 딕셔너리
        save_path: 저장 경로
        plot_type: "type1", "type2", "type3" 중 하나
    """
    original_data = data_dict['original']
    augmented_data = data_dict['augmented']
    
    # Type별 플롯 설정
    if plot_type == "type1":
        # Type 1: 원본 입력 + 증강된 입력 + 증강된 입력의 특징 벡터
        plot_data = [
            ("Original Input", original_data['input_data'], original_data['target_labels']),
            ("Augmented Input", augmented_data['input_data'], augmented_data['target_labels']),
            ("Augmented Features", augmented_data['features'], augmented_data['target_labels'])
        ]
        title_suffix = "Original vs Augmented Input + Aug Features"
    elif plot_type == "type2":
        # Type 2: 원본 입력 + 원본 입력의 특징 벡터
        plot_data = [
            ("Original Input", original_data['input_data'], original_data['target_labels']),
            ("Original Features", original_data['features'], original_data['target_labels'])
        ]
        title_suffix = "Original Input vs Features"
    elif plot_type == "type3":
        # Type 3: 증강된 입력 + 증강된 입력의 특징 벡터
        plot_data = [
            ("Augmented Input", augmented_data['input_data'], augmented_data['target_labels']),
            ("Augmented Features", augmented_data['features'], augmented_data['target_labels'])
        ]
        title_suffix = "Augmented Input vs Features"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")
    
    n_plots = len(plot_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    sns.set_theme(style="white", context="notebook")
    
    # 공통 색상 설정
    unique_labels = np.unique(original_data['target_labels'])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for idx, (name, data, labels) in enumerate(plot_data):
        # 데이터 전처리
        if len(data.shape) > 2:  # 입력 데이터인 경우
            data_flat = data.reshape(data.shape[0], -1)
        else:  # 특징 데이터인 경우
            data_flat = data
        
        # 표준화
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_flat)
        
        # t-SNE
        perplexity_val = min(30, data.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, 
                   max_iter=1000, init='pca', learning_rate='auto')
        data_tsne = tsne.fit_transform(data_scaled)
        
        # 플롯
        ax = axes[idx]
        for label in unique_labels:
            mask = labels == label
            ax.scatter(data_tsne[mask, 0], data_tsne[mask, 1], 
                      c=[color_map[label]], label=f'Class {label}',
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
        
        ax.set_title(f'{name} t-SNE', fontweight='bold')
        ax.legend(frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    
    plt.suptitle(f'{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        filename = f"augmentation_comparison_{plot_type}"
        plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, f"{filename}.svg"), format='svg', bbox_inches='tight')
        print(f"💾 저장: {filename}")
    
    plt.show()


def analyze_augmentation_effects(npz_file_original: str, npz_file_augmented: str, save_path: Optional[str] = None):
    """데이터 증강 효과 종합 분석"""
    print("🔍 데이터 증강 효과 분석 시작")
    
    # 데이터 로드
    orig_data = load_extracted_features(npz_file_original)
    aug_data = load_extracted_features(npz_file_augmented)
    
    combined_data = {
        'original': orig_data,
        'augmented': aug_data
    }
    
    # 3가지 타입 모두 생성
    for plot_type in ["type1", "type2", "type3"]:
        print(f"\n📊 {plot_type.upper()} 시각화 생성")
        plot_augmentation_comparison_tsne(combined_data, save_path, plot_type)


def extract_and_visualize_augmentation(analysis_result_path: str, test_config_base_path: str, 
                                     test_data_default_path: str, output_dir: str,
                                     data_augmentation_config_path: str, max_extractions: int = 1):
    """증강 데이터 추출 및 시각화 통합 함수"""
    from .checkpoint_feature_extractor import CheckpointFeatureRunner, FeatureExtractor
    
    print("🚀 데이터 증강 추출 및 시각화 시작")
    
    # 체크포인트 정보 로드
    import pandas as pd
    df = pd.read_csv(analysis_result_path)
    valid_df = df[df['checkpoint_found'] == True].head(max_extractions)
    
    if len(valid_df) == 0:
        print("❌ 유효한 체크포인트가 없습니다.")
        return
    
    extractor = FeatureExtractor(test_data_default_path)
    
    for _, checkpoint_info in valid_df.iterrows():
        subject_name = checkpoint_info['test_subject_name']
        checkpoint_path = checkpoint_info['checkpoint_path']
        
        print(f"\n🎯 {subject_name} 처리 중...")
        
        try:
            # 모델 로드
            from .checkpoint_feature_extractor import EEGNetLNL  # 적절한 모델 import
            model = extractor.load_model_from_checkpoint(checkpoint_path, "EEGNetLNL")
            
            # 원본 데이터 모듈
            data_module_orig = extractor.create_data_module(
                f"{test_config_base_path}/{subject_name}.json", 16, None)
            
            # 증강 데이터 모듈  
            da_config = extractor.create_data_augmentation_config(data_augmentation_config_path)
            data_module_aug = extractor.create_data_module(
                f"{test_config_base_path}/{subject_name}.json", 16, da_config)
            
            # 특징 추출
            aug_data = extractor.extract_augmented_features(model, data_module_orig, data_module_aug)
            
            # 시각화
            subject_output_dir = os.path.join(output_dir, subject_name)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            # 3가지 타입 시각화
            for plot_type in ["type1", "type2", "type3"]:
                plot_augmentation_comparison_tsne(aug_data, subject_output_dir, plot_type)
            
        except Exception as e:
            print(f"❌ {subject_name} 처리 실패: {e}")
            continue


def analyze_augmentation_experiments_combined(base_dir: str, save_path: Optional[str] = None):
    """실험별로 모든 개체의 데이터 증강 결과를 결합하여 분석"""
    
    # 실험 디렉토리들 찾기 (analyzing_result가 있는 폴더들)
    experiment_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'analyzing_result' in dirs:
            analyzing_result_path = os.path.join(root, 'analyzing_result')
            augmentation_path = os.path.join(analyzing_result_path, 'augmentation_visualization')
            if os.path.exists(augmentation_path):
                experiment_dirs.append(analyzing_result_path)
    
    if not experiment_dirs:
        print(f"❌ 데이터 증강 시각화 디렉토리를 찾을 수 없습니다: {base_dir}")
        return
    
    print(f"🔍 발견된 실험: {len(experiment_dirs)}개")
    
    for exp_dir in experiment_dirs:
        experiment_name = os.path.basename(os.path.dirname(exp_dir))
        print(f"\n📊 실험 처리 중: {experiment_name}")
        
        try:
            plot_all_subjects_augmentation_combined(exp_dir, save_path)
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            continue
    
    print(f"\n✅ 모든 데이터 증강 실험 분석 완료!")


def plot_all_subjects_augmentation_combined(analyzing_result_dir: str, save_path: Optional[str] = None):
    """한 실험의 모든 개체 데이터 증강 결과를 결합하여 시각화"""
    
    augmentation_dir = os.path.join(analyzing_result_dir, 'augmentation_visualization')
    
    if not os.path.exists(augmentation_dir):
        print(f"❌ 데이터 증강 시각화 폴더가 없습니다: {augmentation_dir}")
        return
    
    # 개체별 폴더 찾기
    subject_dirs = [d for d in os.listdir(augmentation_dir) 
                   if os.path.isdir(os.path.join(augmentation_dir, d))]
    
    if not subject_dirs:
        print(f"❌ 개체 폴더가 없습니다: {augmentation_dir}")
        return
    
    print(f"🔍 발견된 개체: {len(subject_dirs)}개 - {subject_dirs}")
    
    # 각 개체별로 데이터 로드 및 결합
    combined_data = {
        'type1': {'original_input': [], 'augmented_input': [], 'augmented_features': [], 'labels': [], 'subjects': []},
        'type2': {'original_input': [], 'original_features': [], 'labels': [], 'subjects': []},
        'type3': {'augmented_input': [], 'augmented_features': [], 'labels': [], 'subjects': []}
    }
    
    subject_loaded = []
    
    for subject_name in subject_dirs:
        subject_path = os.path.join(augmentation_dir, subject_name)
        
        # 각 개체의 데이터 로드 (NPZ 파일이나 기타 저장된 데이터)
        try:
            # 실제로는 데이터 증강 결과를 어딘가에 저장해야 함
            # 현재는 예시로 NPZ 파일이 있다고 가정
            orig_npz_pattern = os.path.join(subject_path, "*_original_features.npz")
            aug_npz_pattern = os.path.join(subject_path, "*_augmented_features.npz")
            
            orig_files = glob.glob(orig_npz_pattern)
            aug_files = glob.glob(aug_npz_pattern)
            
            if not orig_files or not aug_files:
                print(f"⚠️ {subject_name}: 필요한 NPZ 파일이 없습니다")
                continue
            
            # 데이터 로드
            orig_data = np.load(orig_files[0])
            aug_data = np.load(aug_files[0])
            
            # Type 1 데이터 결합
            combined_data['type1']['original_input'].append(orig_data['input_data'])
            combined_data['type1']['augmented_input'].append(aug_data['input_data'])
            combined_data['type1']['augmented_features'].append(aug_data['features'])
            combined_data['type1']['labels'].append(orig_data['target_labels'])
            combined_data['type1']['subjects'].extend([subject_name] * len(orig_data['target_labels']))
            
            # Type 2 데이터 결합
            combined_data['type2']['original_input'].append(orig_data['input_data'])
            combined_data['type2']['original_features'].append(orig_data['features'])
            combined_data['type2']['labels'].append(orig_data['target_labels'])
            combined_data['type2']['subjects'].extend([subject_name] * len(orig_data['target_labels']))
            
            # Type 3 데이터 결합
            combined_data['type3']['augmented_input'].append(aug_data['input_data'])
            combined_data['type3']['augmented_features'].append(aug_data['features'])
            combined_data['type3']['labels'].append(aug_data['target_labels'])
            combined_data['type3']['subjects'].extend([subject_name] * len(aug_data['target_labels']))
            
            subject_loaded.append(subject_name)
            
        except Exception as e:
            print(f"⚠️ {subject_name} 로드 실패: {str(e)}")
            continue
    
    if not subject_loaded:
        print("❌ 로드된 개체가 없습니다")
        return
    
    print(f"✅ 로드된 개체: {len(subject_loaded)}개 - {subject_loaded}")
    
    # 각 타입별로 시각화 생성
    experiment_name = os.path.basename(os.path.dirname(analyzing_result_dir))
    
    for plot_type in ['type1', 'type2', 'type3']:
        try:
            plot_combined_augmentation_tsne(combined_data[plot_type], plot_type, 
                                          experiment_name, subject_loaded, 
                                          analyzing_result_dir if save_path is None else save_path)
        except Exception as e:
            print(f"❌ {plot_type} 시각화 실패: {str(e)}")
            continue


def plot_combined_augmentation_tsne(data_dict: Dict, plot_type: str, experiment_name: str, 
                                   subjects: List[str], save_path: str):
    """결합된 데이터로 증강 t-SNE 시각화"""
    
    # 데이터 결합
    if plot_type == 'type1':
        # Type 1: 원본 입력 + 증강된 입력 + 증강된 특징
        orig_input = np.vstack(data_dict['original_input'])
        aug_input = np.vstack(data_dict['augmented_input'])
        aug_features = np.vstack(data_dict['augmented_features'])
        labels = np.hstack(data_dict['labels'])
        subjects_list = data_dict['subjects']
        
        plot_data = [
            ("Original Input", orig_input, labels),
            ("Augmented Input", aug_input, labels),
            ("Augmented Features", aug_features, labels)
        ]
        title_suffix = "Original vs Augmented Input + Aug Features"
        
    elif plot_type == 'type2':
        # Type 2: 원본 입력 + 원본 특징
        orig_input = np.vstack(data_dict['original_input'])
        orig_features = np.vstack(data_dict['original_features'])
        labels = np.hstack(data_dict['labels'])
        subjects_list = data_dict['subjects']
        
        plot_data = [
            ("Original Input", orig_input, labels),
            ("Original Features", orig_features, labels)
        ]
        title_suffix = "Original Input vs Features"
        
    elif plot_type == 'type3':
        # Type 3: 증강된 입력 + 증강된 특징
        aug_input = np.vstack(data_dict['augmented_input'])
        aug_features = np.vstack(data_dict['augmented_features'])
        labels = np.hstack(data_dict['labels'])
        subjects_list = data_dict['subjects']
        
        plot_data = [
            ("Augmented Input", aug_input, labels),
            ("Augmented Features", aug_features, labels)
        ]
        title_suffix = "Augmented Input vs Features"
    
    # 시각화 생성
    n_plots = len(plot_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 8))
    if n_plots == 1:
        axes = [axes]
    
    sns.set_theme(style="white", context="notebook")
    
    # 공통 색상 설정 (클래스별)
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 개체별 마커 설정
    unique_subjects = list(set(subjects_list))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][: len(unique_subjects)]
    marker_map = {subj: markers[i] for i, subj in enumerate(unique_subjects)}
    
    for idx, (name, data, plot_labels) in enumerate(plot_data):
        # 데이터 전처리
        if len(data.shape) > 2:  # 입력 데이터인 경우
            data_flat = data.reshape(data.shape[0], -1)
        else:  # 특징 데이터인 경우
            data_flat = data
        
        # 표준화
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_flat)
        
        # t-SNE
        perplexity_val = min(30, data.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, 
                   max_iter=1000, init='pca', learning_rate='auto')
        data_tsne = tsne.fit_transform(data_scaled)
        
        # 플롯
        ax = axes[idx]
        
        # 클래스별, 개체별로 플롯
        for label in unique_labels:
            for subject in unique_subjects:
                # 해당 클래스와 개체에 속하는 포인트들 찾기
                label_mask = plot_labels == label
                subject_mask = np.array([s == subject for s in subjects_list])
                mask = label_mask & subject_mask
                
                if np.sum(mask) > 0:
                    ax.scatter(data_tsne[mask, 0], data_tsne[mask, 1],
                             c=[color_map[label]], marker=marker_map[subject],
                             label=f'{subject}-Class{label}', alpha=0.7, s=60,
                             edgecolors='black', linewidth=0.3)
        
        ax.set_title(f'{name} t-SNE\n({len(unique_subjects)} subjects combined)', 
                    fontweight='bold', fontsize=12)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    
    plt.suptitle(f'{experiment_name} - {title_suffix}\nCombined Analysis ({len(unique_subjects)} subjects)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    filename = f"augmentation_combined_{plot_type}_{experiment_name}"
    plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f"{filename}.svg"), format='svg', bbox_inches='tight')
    print(f"💾 저장: {filename}")
    
    plt.show()


    plt.show()


if __name__ == "__main__":
    # 테스트 실행
    base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60"
    save_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/visualization_results"
    
    analyze_all_extracted_features(base_dir, save_dir)
