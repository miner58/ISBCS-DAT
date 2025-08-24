# 1. 환경 설정 및 라이브러리 임포트
import os
import re
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# 한글 폰트 설정
def setup_korean_font():
    """운영체제에 맞는 한글 폰트를 설정합니다."""
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
    elif platform.system() == 'Darwin':  # macOS
        font_name = 'AppleGothic'
    else:  # Linux 등
        font_name = 'NanumGothic'
    
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    print(f"한글 폰트 설정 완료: {font_name}")

# 2. 설정 관리 클래스
@dataclass
class ExperimentConfig:
    """실험 설정을 관리하는 데이터클래스"""
    
    # 기본 경로 설정
    parent_folder_path: str = "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_da_one_60"
    csv_filename: str = "max_metric_per_subject.csv"
    
    # 실험 폴더 목록
    experiment_folders: List[str] = None
    
    # 조건 매핑 (정규표현식 패턴)
    condition_patterns: Dict[str, str] = None
    
    # 메트릭 설정
    metrics: List[str] = None
    metric_labels: List[str] = None
    
    # 시각화 설정
    colors: List[str] = None
    font_sizes: Dict[str, int] = None
    
    # 컬럼 매핑
    column_mapping: Dict[str, str] = None
    required_columns: List[str] = None
    
    # 🆕 Baseline 별도 지정 기능
    fixed_baseline: bool = False
    baseline_parent_path: str = None
    baseline_condition_key: str = "baseline"  # conditions에서 baseline을 가리키는 키
    
    def __post_init__(self):
        """기본값 설정"""
        if self.experiment_folders is None:
            self.experiment_folders = [
                "1_inner_wireless",
                "2_inner_wire", 
                "3_wireless2wire",
                "4_wire2wireless"
            ]
        
        if self.condition_patterns is None:
            self.condition_patterns = {
                "w/ LNL": r"_LNL_",
                "w/o LNL": r"eau_b"
            }
        
        if self.metrics is None:
            self.metrics = ["macro_accuracy", "macro_precision", "macro_recall", "macro_f1"]
        
        if self.metric_labels is None:
            self.metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        
        if self.colors is None:
            self.colors = ["#4169e1", "#dc143c"]
        
        if self.font_sizes is None:
            self.font_sizes = {"title": 26, "tick": 16, "table": 14}
        
        if self.column_mapping is None:
            self.column_mapping = {
                'config/train_loop_config/grl_lambda': 'grl_lambda',
                'config/train_loop_config/subject_name': 'subject_name',
                'test/report/accuracy': 'accuracy',
                'test/report/macro avg/precision': 'macro_precision',
                'test/report/macro avg/recall': 'macro_recall',
                'test/report/macro avg/f1-score': 'macro_f1',
                'config/train_loop_config/lr': 'lr',
                'test/report/macro avg/accuracy': 'macro_accuracy',
            }
        
        if self.required_columns is None:
            self.required_columns = [
                'config/train_loop_config/lr',
                'test_CM/0_0', 'test_CM/0_1', 'test_CM/1_0', 'test_CM/1_1',
                'test/report/macro avg/accuracy', 'test/report/accuracy',
                'test/report/macro avg/precision', 'test/report/macro avg/recall',
                'test/report/macro avg/f1-score', 'test_subject_name'
            ]

# 2-1. 다중 실험 비교 설정 클래스
@dataclass
class MultiExperimentConfig:
    """여러 실험을 하나의 그래프에서 비교하기 위한 설정"""
    
    # 실험 정의 - 각 실험의 경로, 조건, 라벨 정보
    experiments: List[Dict[str, Any]] = None
    
    # 공통 설정
    experiment_folders: List[str] = None
    csv_filename: str = "max_metric_per_subject.csv"
    
    # 메트릭 설정
    metrics: List[str] = None
    metric_labels: List[str] = None
    
    # 시각화 설정
    colors: List[str] = None
    font_sizes: Dict[str, int] = None
    
    # 컬럼 매핑
    column_mapping: Dict[str, str] = None
    required_columns: List[str] = None
    
    def __post_init__(self):
        """기본값 설정"""
        if self.experiment_folders is None:
            self.experiment_folders = [
                "1_inner_wireless",
                "2_inner_wire", 
                "3_wireless2wire",
                "4_wire2wireless"
            ]
        
        if self.experiments is None:
            self.experiments = [
                {
                    "name": "baseline",
                    "path": "/path/to/baseline",
                    "condition_pattern": r"eau_b",
                    "label": "Baseline",
                    "color": "#4169e1"
                },
                {
                    "name": "method1",
                    "path": "/path/to/method1",
                    "condition_pattern": r"_LNL_",
                    "label": "w/ LNL",
                    "color": "#dc143c"
                }
            ]
        
        if self.metrics is None:
            self.metrics = ["macro_accuracy", "macro_precision", "macro_recall", "macro_f1"]
        
        if self.metric_labels is None:
            self.metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        
        if self.colors is None:
            # 더 많은 색상 지원
            self.colors = [
                "#4169e1",  # Royal Blue
                "#dc143c",  # Crimson
                "#32cd32",  # Lime Green
                "#ff8c00",  # Dark Orange
                "#9932cc",  # Dark Orchid
                "#00ced1",  # Dark Turquoise
                "#ff69b4",  # Hot Pink
                "#8b4513"   # Saddle Brown
            ]
        
        if self.font_sizes is None:
            self.font_sizes = {"title": 26, "tick": 16, "table": 14}
        
        if self.column_mapping is None:
            self.column_mapping = {
                'config/train_loop_config/grl_lambda': 'grl_lambda',
                'config/train_loop_config/subject_name': 'subject_name',
                'test/report/accuracy': 'accuracy',
                'test/report/macro avg/precision': 'macro_precision',
                'test/report/macro avg/recall': 'macro_recall',
                'test/report/macro avg/f1-score': 'macro_f1',
                'config/train_loop_config/lr': 'lr',
                'test/report/macro avg/accuracy': 'macro_accuracy',
            }
        
        if self.required_columns is None:
            self.required_columns = [
                'config/train_loop_config/lr',
                'test_CM/0_0', 'test_CM/0_1', 'test_CM/1_0', 'test_CM/1_1',
                'test/report/macro avg/accuracy', 'test/report/accuracy',
                'test/report/macro avg/precision', 'test/report/macro avg/recall',
                'test/report/macro avg/f1-score', 'test_subject_name'
            ]

# 3. 데이터 처리 클래스
class DataProcessor:
    """실험 데이터를 처리하는 클래스"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """필요한 컬럼만 필터링합니다."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            print(f"[경고] 누락된 컬럼: {missing}")
        
        available_columns = [c for c in self.config.required_columns if c in df.columns]
        return df[available_columns]
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명을 표준화합니다."""
        return df.rename(columns=self.config.column_mapping)
    
    def load_experiment_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """모든 실험 데이터를 로드하고 조건별로 분류합니다."""
        data_dict = {}
        
        # 조건 분류: baseline과 일반 조건 분리
        baseline_conditions, regular_conditions = self._classify_conditions()
        
        # 실험 폴더 경로 생성
        folder_paths = self._get_experiment_folder_paths()
        
        for exp_name, exp_path in zip(self.config.experiment_folders, folder_paths):
            data_dict[exp_name] = {key: None for key in self.config.condition_patterns.keys()}
            print(f"\\n실험: {exp_name}")
            print(f"경로: {exp_path}")
            
            if not os.path.isdir(exp_path):
                print(f"[오류] 경로가 없습니다: {exp_path}")
                continue
            
            # 1. 일반 조건 데이터 로드 (기존 parent_path)
            if regular_conditions:
                print(f"  📂 일반 조건 데이터 로드 중...")
                regular_data = self._load_data_from_path(exp_path, regular_conditions, exp_name)
                data_dict[exp_name].update(regular_data)
            
            # 2. baseline 조건 데이터 로드 (별도 baseline_parent_path)
            if baseline_conditions and self.config.fixed_baseline:
                if self.config.baseline_parent_path:
                    baseline_exp_path = os.path.join(self.config.baseline_parent_path, exp_name)
                    print(f"  📂 Baseline 데이터 로드 중 (별도 경로: {baseline_exp_path})")
                    baseline_data = self._load_data_from_path(baseline_exp_path, baseline_conditions, exp_name)
                    data_dict[exp_name].update(baseline_data)
                else:
                    print(f"  ⚠️ fixed_baseline=True이지만 baseline_parent_path가 설정되지 않았습니다.")
            elif baseline_conditions and not self.config.fixed_baseline:
                # 기존 방식: 같은 경로에서 baseline 로드
                print(f"  📂 Baseline 데이터 로드 중 (동일 경로)")
                baseline_data = self._load_data_from_path(exp_path, baseline_conditions, exp_name)
                data_dict[exp_name].update(baseline_data)
                    
        return data_dict
    
    def _classify_conditions(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """조건을 baseline과 일반 조건으로 분류합니다."""
        baseline_conditions = []
        regular_conditions = []
        
        for condition_name, pattern in self.config.condition_patterns.items():
            if (self.config.fixed_baseline and 
                condition_name == self.config.baseline_condition_key):
                baseline_conditions.append((condition_name, pattern))
            else:
                regular_conditions.append((condition_name, pattern))
        
        return baseline_conditions, regular_conditions
    
    def _load_data_from_path(self, base_path: str, conditions: List[Tuple[str, str]], exp_name: str) -> Dict[str, pd.DataFrame]:
        """지정된 경로에서 조건에 맞는 데이터를 로드합니다."""
        data_dict = {}
        condition_patterns = {key: re.compile(pattern) for key, pattern in conditions}
        
        if not os.path.isdir(base_path):
            print(f"    [경고] 경로가 없습니다: {base_path}")
            return data_dict
                
        # 서브폴더 탐색
        for sub_folder in os.listdir(base_path):
            sub_analyze_path = os.path.join(base_path, sub_folder, 'analyzing_result')
            if not os.path.isdir(sub_analyze_path):
                continue
                
            csv_path = os.path.join(sub_analyze_path, self.config.csv_filename)
            if not os.path.isfile(csv_path):
                print(f"    [경고] 파일 없음: {csv_path}")
                continue
            
            # 데이터 로드 및 전처리
            try:
                df = pd.read_csv(csv_path)
                df = self.filter_columns(df)
                df = self.rename_columns(df)
                
                # 조건별 데이터 할당
                for condition_name, pattern in condition_patterns.items():
                    if pattern.search(sub_folder):
                        print(f"    조건: {condition_name}, 서브폴더: {sub_folder}")
                        data_dict[condition_name] = df
                        break
                        
            except Exception as e:
                print(f"    [오류] 데이터 로드 실패 ({csv_path}): {e}")
                
        return data_dict
    
    def _get_experiment_folder_paths(self) -> List[str]:
        """실험 폴더 경로 목록을 생성합니다."""
        if not os.path.isdir(self.config.parent_folder_path):
            raise ValueError(f"부모 폴더가 존재하지 않습니다: {self.config.parent_folder_path}")
        
        available_folders = [
            folder for folder in os.listdir(self.config.parent_folder_path) 
            if folder in self.config.experiment_folders
        ]
        
        return sorted(
            [os.path.join(self.config.parent_folder_path, folder) for folder in available_folders],
            key=lambda x: self.config.experiment_folders.index(os.path.basename(x))
        )

# 3-1. 다중 실험 데이터 처리 클래스
class MultiExperimentDataProcessor:
    """여러 실험의 데이터를 처리하고 통합하는 클래스"""
    
    def __init__(self, config: MultiExperimentConfig):
        self.config = config
        
    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """필요한 컬럼만 필터링합니다."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            print(f"[경고] 누락된 컬럼: {missing}")
        
        available_columns = [c for c in self.config.required_columns if c in df.columns]
        return df[available_columns]
    
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명을 표준화합니다."""
        return df.rename(columns=self.config.column_mapping)
    
    def load_single_experiment_data(self, experiment_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """단일 실험의 데이터를 로드합니다."""
        experiment_data = {}
        
        for exp_folder in self.config.experiment_folders:
            print(f"\\n  📂 {experiment_config['name']} - {exp_folder} 로드 중...")
            
            # 실험 폴더 경로 생성
            exp_path = os.path.join(experiment_config["path"], exp_folder)
            
            if not os.path.isdir(exp_path):
                print(f"    [경고] 경로가 없습니다: {exp_path}")
                experiment_data[exp_folder] = None
                continue
            
            # 조건에 맞는 서브폴더 찾기
            pattern = re.compile(experiment_config["condition_pattern"])
            found_data = None
            
            for sub_folder in os.listdir(exp_path):
                if pattern.search(sub_folder):
                    sub_analyze_path = os.path.join(exp_path, sub_folder, 'analyzing_result')
                    csv_path = os.path.join(sub_analyze_path, self.config.csv_filename)
                    
                    if os.path.isfile(csv_path):
                        try:
                            df = pd.read_csv(csv_path)
                            df = self.filter_columns(df)
                            df = self.rename_columns(df)
                            found_data = df
                            print(f"    ✅ 데이터 로드 완료: {sub_folder}")
                            break
                        except Exception as e:
                            print(f"    [오류] 데이터 로드 실패 ({csv_path}): {e}")
            
            experiment_data[exp_folder] = found_data
            if found_data is None:
                print(f"    ❌ 조건에 맞는 데이터를 찾을 수 없습니다")
        
        return experiment_data
    
    def load_all_experiments_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """모든 실험의 데이터를 로드합니다."""
        all_data = {}
        
        print("=== 다중 실험 데이터 로드 시작 ===")
        
        for experiment in self.config.experiments:
            exp_name = experiment["name"]
            print(f"\\n🔄 실험 로드: {exp_name} ({experiment['label']})")
            print(f"   경로: {experiment['path']}")
            print(f"   패턴: {experiment['condition_pattern']}")
            
            experiment_data = self.load_single_experiment_data(experiment)
            all_data[exp_name] = experiment_data
        
        print("\\n=== 다중 실험 데이터 로드 완료 ===")
        return all_data
    
    def aggregate_experiments_for_comparison(self, all_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """실험 폴더별로 데이터를 재구성하여 비교가 쉽도록 합니다."""
        aggregated_data = {}
        
        for exp_folder in self.config.experiment_folders:
            aggregated_data[exp_folder] = {}
            
            for experiment in self.config.experiments:
                exp_name = experiment["name"]
                exp_label = experiment["label"]
                
                if exp_name in all_data and exp_folder in all_data[exp_name]:
                    aggregated_data[exp_folder][exp_label] = all_data[exp_name][exp_folder]
                else:
                    aggregated_data[exp_folder][exp_label] = None
        
        return aggregated_data

# 4. 시각화 클래스
class ExperimentVisualizer:
    """실험 결과를 시각화하는 클래스"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def plot_confusion_matrix(self, cm: List[List[float]], subject_name: str, save_path: str):
        """Confusion Matrix를 플롯합니다."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {subject_name}')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def create_experiment_comparison_plot(self, 
                                        data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                        experiment_name: str,
                                        output_dir: str) -> None:
        """실험 비교 그래프를 생성합니다."""
        
        conditions = list(self.config.condition_patterns.keys())
        condition_labels = list(conditions)
        
        # 데이터 확인
        dfs = [data_dict.get(experiment_name, {}).get(cond) for cond in conditions]
        if any(df is None for df in dfs):
            print(f"[경고] '{experiment_name}'에 필요한 데이터가 없습니다. 건너뜁니다.")
            return
            
        # 통계 계산
        means = [df[self.config.metrics].mean().values for df in dfs]
        stds = [df[self.config.metrics].std().values for df in dfs]
        
        # 그래프 생성
        x = np.arange(len(self.config.metrics))
        width = 0.8 / len(conditions)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, (m, s, lbl) in enumerate(zip(means, stds, condition_labels)):
            offset = (i - (len(conditions)-1)/2) * width
            color = self.config.colors[i] if i < len(self.config.colors) else None
            
            ax.bar(x + offset, m, width, yerr=s, capsize=5, label=lbl, color=color)
            
            # 개별 데이터 포인트 표시
            for xi, metric in enumerate(self.config.metrics):
                ax.plot([x[xi] + offset] * len(dfs[i]),
                       dfs[i][metric], 'o', color='black', alpha=0.6)
        
        # 그래프 스타일링
        ax.set_title(experiment_name, fontsize=self.config.font_sizes["title"], pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.config.metric_labels, fontsize=self.config.font_sizes["tick"])
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=self.config.font_sizes["table"])
        
        # 테이블 데이터 생성
        table_data, csv_data = self._create_table_data(means, stds, condition_labels)
        
        # CSV 저장
        self._save_csv_data(csv_data, experiment_name, output_dir)
        
        # 테이블 추가
        self._add_table_to_plot(ax, table_data)
        
        # 저장 및 표시
        plt.subplots_adjust(bottom=0.2)
        save_path = os.path.join(output_dir, f"{experiment_name}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def _create_table_data(self, means: List[np.ndarray], stds: List[np.ndarray], 
                          condition_labels: List[str]) -> Tuple[List[List[str]], List[Dict[str, str]]]:
        """테이블 데이터를 생성합니다."""
        table_data = [condition_labels]
        csv_data = []
        
        for mi in range(len(self.config.metrics)):
            row_data = []
            csv_row = {'Metric': self.config.metric_labels[mi]}
            
            for i in range(len(condition_labels)):
                mean_val = means[i][mi]
                std_val = stds[i][mi]
                formatted_val = f"{mean_val:.3f} ± {std_val:.3f}"
                row_data.append(formatted_val)
                csv_row[condition_labels[i]] = formatted_val
                
            table_data.append(row_data)
            csv_data.append(csv_row)
            
        return table_data, csv_data
    
    def _save_csv_data(self, csv_data: List[Dict[str, str]], experiment_name: str, output_dir: str):
        """CSV 데이터를 저장합니다."""
        csv_df = pd.DataFrame(csv_data).transpose()
        csv_path = os.path.join(output_dir, f"{experiment_name}_table_data.csv")
        os.makedirs(output_dir, exist_ok=True)
        csv_df.to_csv(csv_path, index=True, header=True)
        print(f"테이블 데이터가 저장되었습니다: {csv_path}")
        
    def _add_table_to_plot(self, ax, table_data: List[List[str]]):
        """플롯에 테이블을 추가합니다."""
        row_labels = [""] + self.config.metric_labels
        table = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            loc="bottom",
            cellLoc="center",
            bbox=[0, -0.35, 1, 0.3]
        )
        
        for cell in table.get_celld().values():
            cell.get_text().set_fontsize(self.config.font_sizes["table"])

# 4-1. 다중 실험 시각화 클래스
class MultiExperimentVisualizer:
    """여러 실험을 하나의 그래프에서 비교하는 시각화 클래스"""
    
    def __init__(self, config: MultiExperimentConfig):
        self.config = config
        
    def create_multi_experiment_comparison_plot(self, 
                                              aggregated_data: Dict[str, Dict[str, pd.DataFrame]], 
                                              experiment_folder: str,
                                              output_dir: str) -> None:
        """여러 실험을 하나의 그래프에서 비교합니다."""
        
        print(f"\\n🎨 다중 실험 비교 그래프 생성: {experiment_folder}")
        
        # 실험 라벨 목록 가져오기
        experiment_labels = [exp["label"] for exp in self.config.experiments]
        
        # 데이터 확인 및 유효한 데이터만 필터링
        valid_experiments = []
        valid_labels = []
        valid_colors = []
        
        for i, experiment in enumerate(self.config.experiments):
            exp_label = experiment["label"]
            if (experiment_folder in aggregated_data and 
                exp_label in aggregated_data[experiment_folder] and 
                aggregated_data[experiment_folder][exp_label] is not None):
                
                valid_experiments.append(aggregated_data[experiment_folder][exp_label])
                valid_labels.append(exp_label)
                
                # 색상 할당: experiment에 color가 지정되어 있으면 사용, 없으면 기본 색상
                color = experiment.get("color", self.config.colors[i % len(self.config.colors)])
                valid_colors.append(color)
        
        if not valid_experiments:
            print(f"[경고] '{experiment_folder}'에 유효한 실험 데이터가 없습니다.")
            return
        
        print(f"  📊 유효한 실험 수: {len(valid_experiments)}")
        print(f"  📝 실험 라벨: {valid_labels}")
        
        # 통계 계산
        means = [df[self.config.metrics].mean().values for df in valid_experiments]
        stds = [df[self.config.metrics].std().values for df in valid_experiments]
        
        # 그래프 생성
        x = np.arange(len(self.config.metrics))
        width = 0.8 / len(valid_experiments)
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 각 실험에 대해 바 그래프 생성
        for i, (m, s, lbl, color) in enumerate(zip(means, stds, valid_labels, valid_colors)):
            offset = (i - (len(valid_experiments)-1)/2) * width
            
            bars = ax.bar(x + offset, m, width, yerr=s, capsize=5, 
                         label=lbl, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 개별 데이터 포인트 표시 (동일 선상에 위치)
            for xi, metric in enumerate(self.config.metrics):
                values = valid_experiments[i][metric].values
                # 점들을 동일한 x 위치에 표시
                ax.scatter([x[xi] + offset] * len(values),
                          values, color='dimgray', alpha=0.7, s=40, zorder=3)
        
        # 그래프 스타일링
        ax.set_title(f"{experiment_folder}", 
                    fontsize=self.config.font_sizes["title"], pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.config.metric_labels, fontsize=self.config.font_sizes["tick"])
        ax.set_ylim(0, 1)
        
        # 범례를 그래프 오른쪽 밖에 위치
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=self.config.font_sizes["table"], 
                 frameon=True, fancybox=True, shadow=True)
        
        # 격자 추가
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # 테이블 데이터 생성
        table_data, csv_data = self._create_multi_table_data(means, stds, valid_labels)
        
        # CSV 저장
        self._save_multi_csv_data(csv_data, experiment_folder, output_dir)
        
        # 테이블 추가 (메트릭과 정렬)
        self._add_multi_table_to_plot(ax, table_data, x)
        
        # 저장 및 표시
        plt.subplots_adjust(bottom=0.25, right=0.8)
        save_path = os.path.join(output_dir, f"multi_exp_{experiment_folder}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
        
        print(f"  💾 그래프 저장 완료: {save_path}")
    
    def create_all_multi_experiment_plots(self, 
                                        aggregated_data: Dict[str, Dict[str, pd.DataFrame]], 
                                        output_dir: str) -> None:
        """모든 실험 폴더에 대해 다중 실험 비교 그래프를 생성합니다."""
        
        print("\\n🎨 모든 다중 실험 비교 그래프 생성 시작")
        
        for exp_folder in self.config.experiment_folders:
            if exp_folder in aggregated_data:
                self.create_multi_experiment_comparison_plot(aggregated_data, exp_folder, output_dir)
        
        print("\\n✅ 모든 다중 실험 비교 그래프 생성 완료")
    
    def _create_multi_table_data(self, means: List[np.ndarray], stds: List[np.ndarray], 
                               experiment_labels: List[str]) -> Tuple[List[List[str]], List[Dict[str, str]]]:
        """다중 실험 테이블 데이터를 생성합니다. (metrics를 열로 배치)"""
        # 헤더: 첫 번째 행은 메트릭 라벨들
        table_data = [[""] + self.config.metric_labels]
        csv_data = []
        
        # 각 실험(행)에 대해 데이터 생성
        for i, exp_label in enumerate(experiment_labels):
            row_data = [exp_label]  # 첫 번째 열은 실험 라벨
            csv_row = {'Experiment': exp_label}
            
            # 각 메트릭(열)에 대해 mean ± std 추가
            for mi in range(len(self.config.metrics)):
                mean_val = means[i][mi]
                std_val = stds[i][mi]
                formatted_val = f"{mean_val:.3f} ± {std_val:.3f}"
                row_data.append(formatted_val)
                csv_row[self.config.metric_labels[mi]] = formatted_val
                
            table_data.append(row_data)
            csv_data.append(csv_row)
            
        return table_data, csv_data
    
    def _save_multi_csv_data(self, csv_data: List[Dict[str, str]], experiment_folder: str, output_dir: str):
        """다중 실험 CSV 데이터를 저장합니다."""
        csv_df = pd.DataFrame(csv_data).transpose()
        csv_path = os.path.join(output_dir, f"multi_exp_{experiment_folder}_table_data.csv")
        os.makedirs(output_dir, exist_ok=True)
        csv_df.to_csv(csv_path, index=True, header=True)
        print(f"  💾 테이블 데이터 저장: {csv_path}")
        
    def _add_multi_table_to_plot(self, ax, table_data: List[List[str]], x_positions):
        """다중 실험 플롯에 테이블을 추가합니다. (metrics와 정렬)"""
        # x_positions에 맞춰 테이블 열 너비 계산
        num_metrics = len(self.config.metric_labels)
        
        # 첫 번째 열(실험명)은 고정 너비, 나머지는 x_positions에 맞춤
        col_widths = [0.2] + [0.8/num_metrics] * num_metrics
        
        table = ax.table(
            cellText=table_data[1:],  # 첫 번째 행을 제외한 데이터
            colLabels=table_data[0],  # 첫 번째 행을 열 헤더로 사용
            loc="bottom",
            cellLoc="center",
            bbox=[0, -0.4, 1, 0.35],
            colWidths=col_widths
        )
        
        # 테이블 스타일링
        for cell in table.get_celld().values():
            cell.get_text().set_fontsize(self.config.font_sizes["table"])
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)
        
        # 헤더 행 스타일링
        for i in range(len(table_data[0])):
            if (0, i) in table.get_celld():
                table[(0, i)].set_facecolor('#E6E6FA')
                table[(0, i)].set_text_props(weight='bold')

# 5. 실험 실행 관리 클래스
class ExperimentRunner:
    """전체 실험 파이프라인을 실행하는 클래스"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.visualizer = ExperimentVisualizer(config)
        
    def run_complete_analysis(self, save_confusion_matrices: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """전체 분석을 실행합니다."""
        print("=== 실험 분석 시작 ===")
        
        # 1. 데이터 로드
        print("\\n1. 데이터 로드 중...")
        data_dict = self.data_processor.load_experiment_data()
        
        # 2. Confusion Matrix 저장 (옵션)
        if save_confusion_matrices:
            print("\\n2. Confusion Matrix 저장 중...")
            self._save_confusion_matrices(data_dict)
        
        # 3. 비교 그래프 생성
        print("\\n3. 비교 그래프 생성 중...")
        output_dir = os.path.join(self.config.parent_folder_path, 'plots')
        
        for experiment_name in self.config.experiment_folders:
            if experiment_name in data_dict:
                print(f"  - {experiment_name} 그래프 생성")
                self.visualizer.create_experiment_comparison_plot(
                    data_dict, experiment_name, output_dir
                )
        
        print("\\n=== 분석 완료 ===")
        return data_dict
    
    def _save_confusion_matrices(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]):
        """Confusion Matrix를 저장합니다."""
        cm_columns = {'test_subject_name', 'test_CM/0_0', 'test_CM/0_1', 'test_CM/1_0', 'test_CM/1_1'}
        
        for exp_name, conditions in data_dict.items():
            for condition_name, df in conditions.items():
                if df is None:
                    continue
                    
                # 컬럼명 매핑 확인
                available_cm_cols = set()
                for col in df.columns:
                    if 'test_subject_name' in col:
                        available_cm_cols.add('test_subject_name')
                    elif any(cm_col in col for cm_col in ['CM/0_0', 'CM/0_1', 'CM/1_0', 'CM/1_1']):
                        available_cm_cols.add(col)
                
                if len(available_cm_cols) < 5:  # test_subject_name + 4개 CM 컬럼
                    continue
                
                # Confusion Matrix 저장
                for _, row in df.iterrows():
                    try:
                        subject_name = row['test_subject_name']
                        
                        # CM 컬럼 찾기
                        cm_00 = row.get('test_CM/0_0', row.get('CM/0_0', 0))
                        cm_01 = row.get('test_CM/0_1', row.get('CM/0_1', 0))
                        cm_10 = row.get('test_CM/1_0', row.get('CM/1_0', 0))
                        cm_11 = row.get('test_CM/1_1', row.get('CM/1_1', 0))
                        
                        cm = [[cm_00, cm_01], [cm_10, cm_11]]
                        
                        # 저장 경로 생성
                        exp_path = os.path.join(self.config.parent_folder_path, exp_name)
                        out_dir = os.path.join(exp_path, 'confusion_matrices', condition_name)
                        os.makedirs(out_dir, exist_ok=True)
                        
                        save_path = os.path.join(out_dir, f"{subject_name}_cm.png")
                        self.visualizer.plot_confusion_matrix(cm, subject_name, save_path)
                        
                    except Exception as e:
                        print(f"[경고] Confusion Matrix 저장 실패: {e}")
    
    def run_single_experiment(self, experiment_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """단일 실험만 실행합니다."""
        if experiment_name not in self.config.experiment_folders:
            print(f"[오류] 알 수 없는 실험명: {experiment_name}")
            return None
        
        print(f"=== {experiment_name} 실험 분석 ===")
        
        # 임시로 실험 폴더 목록 변경
        original_folders = self.config.experiment_folders.copy()
        self.config.experiment_folders = [experiment_name]
        
        try:
            data_dict = self.run_complete_analysis(save_confusion_matrices=False)
            return data_dict.get(experiment_name)
        finally:
            # 원래 설정 복원
            self.config.experiment_folders = original_folders

# 5-1. 다중 실험 실행 관리 클래스
class MultiExperimentRunner:
    """다중 실험 비교 파이프라인을 실행하는 클래스"""
    
    def __init__(self, config: MultiExperimentConfig):
        self.config = config
        self.data_processor = MultiExperimentDataProcessor(config)
        self.visualizer = MultiExperimentVisualizer(config)
        
    def run_multi_experiment_analysis(self, output_dir: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """다중 실험 비교 분석을 실행합니다."""
        print("\\n" + "="*60)
        print("🚀 다중 실험 비교 분석 시작")
        print("="*60)
        
        # 출력 디렉토리 설정
        if output_dir is None:
            # 첫 번째 실험의 경로를 기준으로 출력 디렉토리 설정
            base_path = self.config.experiments[0]["path"]
            output_dir = os.path.join(os.path.dirname(base_path), 'multi_experiment_plots')
        
        # 1. 모든 실험 데이터 로드
        print("\\n📂 1단계: 데이터 로드")
        all_data = self.data_processor.load_all_experiments_data()
        
        # 2. 데이터 집계 및 재구성
        print("\\n🔄 2단계: 데이터 집계 및 재구성")
        aggregated_data = self.data_processor.aggregate_experiments_for_comparison(all_data)
        
        # 3. 다중 실험 비교 그래프 생성
        print("\\n🎨 3단계: 다중 실험 비교 그래프 생성")
        self.visualizer.create_all_multi_experiment_plots(aggregated_data, output_dir)
        
        # 4. 결과 요약
        self._print_analysis_summary(aggregated_data, output_dir)
        
        print("\\n" + "="*60)
        print("✅ 다중 실험 비교 분석 완료")
        print("="*60)
        
        return aggregated_data
    
    def run_single_folder_analysis(self, experiment_folder: str, output_dir: str = None) -> Optional[Dict[str, pd.DataFrame]]:
        """특정 실험 폴더만 분석합니다."""
        if experiment_folder not in self.config.experiment_folders:
            print(f"[오류] 알 수 없는 실험 폴더: {experiment_folder}")
            return None
        
        print(f"\\n🎯 단일 폴더 분석: {experiment_folder}")
        
        # 임시로 실험 폴더 목록 변경
        original_folders = self.config.experiment_folders.copy()
        self.config.experiment_folders = [experiment_folder]
        
        try:
            aggregated_data = self.run_multi_experiment_analysis(output_dir)
            return aggregated_data.get(experiment_folder)
        finally:
            # 원래 설정 복원
            self.config.experiment_folders = original_folders
    
    def _print_analysis_summary(self, aggregated_data: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
        """분석 결과 요약을 출력합니다."""
        print("\\n📊 분석 결과 요약")
        print("-" * 40)
        
        experiment_labels = [exp["label"] for exp in self.config.experiments]
        
        for exp_folder, exp_data in aggregated_data.items():
            print(f"\\n📁 {exp_folder}:")
            for exp_label in experiment_labels:
                if exp_label in exp_data and exp_data[exp_label] is not None:
                    data_count = len(exp_data[exp_label])
                    print(f"  ✅ {exp_label}: {data_count}개 데이터")
                else:
                    print(f"  ❌ {exp_label}: 데이터 없음")
        
        print(f"\\n💾 결과 저장 위치: {output_dir}")
    
    def validate_experiment_paths(self) -> bool:
        """실험 경로들의 유효성을 검사합니다."""
        print("\\n🔍 실험 경로 유효성 검사")
        print("-" * 30)
        
        all_valid = True
        for experiment in self.config.experiments:
            exp_name = experiment["name"]
            exp_path = experiment["path"]
            
            if os.path.exists(exp_path):
                print(f"  ✅ {exp_name}: {exp_path}")
            else:
                print(f"  ❌ {exp_name}: {exp_path} (경로 없음)")
                all_valid = False
        
        return all_valid

# 8. 유틸리티 함수 및 헬퍼 함수
def quick_analysis(experiment_name: str, config: ExperimentConfig = None):
    """특정 실험만 빠르게 분석합니다."""
    if config is None:
        config = ExperimentConfig()
    
    runner = ExperimentRunner(config)
    return runner.run_single_experiment(experiment_name)

def create_custom_config(
    experiment_folders: List[str] = None,
    condition_patterns: Dict[str, str] = None,
    parent_path: str = None,
    colors: List[str] = None,
    # 🆕 Baseline 별도 지정 관련 파라미터
    fixed_baseline: bool = False,
    baseline_parent_path: str = None,
    baseline_condition_key: str = "baseline"
) -> ExperimentConfig:
    """커스텀 설정을 쉽게 생성합니다."""
    
    custom_config = ExperimentConfig()
    
    if experiment_folders:
        custom_config.experiment_folders = experiment_folders
    if condition_patterns:
        custom_config.condition_patterns = condition_patterns
    if parent_path:
        custom_config.parent_folder_path = parent_path
    if colors:
        custom_config.colors = colors
    
    # 🆕 Baseline 관련 설정
    custom_config.fixed_baseline = fixed_baseline
    if baseline_parent_path:
        custom_config.baseline_parent_path = baseline_parent_path
    custom_config.baseline_condition_key = baseline_condition_key
    
    return custom_config

def print_available_experiments(config: ExperimentConfig = None):
    """사용 가능한 실험 목록을 출력합니다."""
    if config is None:
        config = ExperimentConfig()
    
    print("=== 사용 가능한 실험 ===")
    for i, exp in enumerate(config.experiment_folders, 1):
        print(f"{i}. {exp}")
    
    print("\\n=== 분석 조건 ===")
    for condition, pattern in config.condition_patterns.items():
        print(f"- {condition}: {pattern}")

def export_results_to_excel(data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                           output_path: str = "experiment_results.xlsx"):
    """결과를 Excel 파일로 내보냅니다."""
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for exp_name, conditions in data_dict.items():
                for condition_name, df in conditions.items():
                    if df is not None:
                        sheet_name = f"{exp_name}_{condition_name}"[:31]  # Excel 시트명 길이 제한
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"결과가 Excel 파일로 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"Excel 저장 중 오류 발생: {e}")

# 8-1. 다중 실험 비교 헬퍼 함수들

def create_multi_experiment_config(
    experiments: List[Dict[str, Any]],
    experiment_folders: List[str] = None,
    colors: List[str] = None,
    output_name: str = "multi_experiment_comparison"
) -> MultiExperimentConfig:
    """다중 실험 설정을 쉽게 생성합니다."""
    
    config = MultiExperimentConfig()
    config.experiments = experiments
    
    if experiment_folders:
        config.experiment_folders = experiment_folders
    if colors:
        config.colors = colors
    
    return config

def run_multi_experiment_comparison(
    experiments: List[Dict[str, Any]],
    experiment_folders: List[str] = None,
    output_dir: str = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """간편한 다중 실험 비교 실행 함수"""
    
    # 설정 생성
    config = create_multi_experiment_config(experiments, experiment_folders)
    
    # 실행기 생성 및 실행
    runner = MultiExperimentRunner(config)
    
    # 경로 유효성 검사
    if not runner.validate_experiment_paths():
        print("⚠️ 일부 실험 경로가 유효하지 않습니다. 계속 진행합니다.")
    
    # 분석 실행
    return runner.run_multi_experiment_analysis(output_dir)

def create_experiment_definition(
    name: str,
    path: str, 
    condition_pattern: str,
    label: str,
    color: str = None
) -> Dict[str, Any]:
    """실험 정의를 쉽게 생성하는 헬퍼 함수"""
    
    experiment = {
        "name": name,
        "path": path,
        "condition_pattern": condition_pattern,
        "label": label
    }
    
    if color:
        experiment["color"] = color
    
    return experiment

def print_multi_experiment_template():
    """다중 실험 비교 사용 예시를 출력합니다."""
    
    print("\\n" + "="*60)
    print("🎯 다중 실험 비교 사용법")
    print("="*60)
    
    print("""
# 1️⃣ 실험 정의 생성
experiments = [
    create_experiment_definition(
        name="baseline",
        path="/path/to/baseline/results",
        condition_pattern=r"eau_b",
        label="Baseline (EEGNet)",
        color="#4169e1"
    ),
    create_experiment_definition(
        name="lnl_method",
        path="/path/to/lnl/results", 
        condition_pattern=r"_LNL_",
        label="w/ LNL",
        color="#dc143c"
    ),
    create_experiment_definition(
        name="channel_norm",
        path="/path/to/channel_norm/results",
        condition_pattern=r"_channelNorm",
        label="w/ ChannelNorm",
        color="#32cd32"
    ),
    create_experiment_definition(
        name="our_method",
        path="/path/to/our_method/results",
        condition_pattern=r"_our_pattern",
        label="Ours",
        color="#ff8c00"
    )
]

# 2️⃣ 실행
results = run_multi_experiment_comparison(
    experiments=experiments,
    experiment_folders=["1_inner_wireless", "2_inner_wire", "3_wireless2wire", "4_wire2wireless"],
    output_dir="/path/to/output"
)
""")
    
    print("="*60)

# 9. 배치 처리 템플릿 - 여러 폴더 한번에 처리

def batch_analysis_template(batch_configs: Optional[List[Dict[str, Any]]] = None, 
                            common_experiments: Optional[List[str]] = None) -> Dict[str, Any]:
    """여러 실험 폴더를 한번에 처리하는 템플릿"""
    
    # ================================
    # 🎯 설정 섹션 - 여기만 수정하세요!
    # ================================
    
    # 처리할 폴더 설정들
    batch_configs = batch_configs
    if batch_configs is None:
        raise ValueError("배치 설정이 필요합니다. 'batch_configs' 인자를 제공하세요.")
    # 공통 실험 폴더 설정
    common_experiments = common_experiments
    if common_experiments is None:
        raise ValueError("공통 실험 폴더가 필요합니다. 'common_experiments' 인자를 제공하세요.")
    
    # ================================
    # 🚀 실행 섹션 - 자동 처리
    # ================================
    
    results = {}
    
    for i, config_info in enumerate(batch_configs, 1):
        print(f"\\n{'='*60}")
        print(f"🔄 배치 {i}/{len(batch_configs)}: {config_info['name']}")
        print(f"{'='*60}")
        
        try:
            # 설정 생성
            config = create_custom_config(
                experiment_folders=common_experiments,
                condition_patterns=config_info["conditions"],
                parent_path=config_info["parent_path"],
                # 🆕 Baseline 별도 지정 지원
                fixed_baseline=config_info.get("fixed_baseline", False),
                baseline_parent_path=config_info.get("baseline_parent_path"),
                baseline_condition_key=config_info.get("baseline_condition_key", "baseline")
            )
            
            # 폴더 존재 여부 확인
            if not os.path.exists(config_info["parent_path"]):
                print(f"⚠️  경로가 존재하지 않습니다: {config_info['parent_path']}")
                results[config_info['name']] = None
                continue
            
            # 분석 실행
            data_dict = main(config)
            results[config_info['name']] = data_dict
            
            print(f"✅ {config_info['name']} 완료!")
            
        except Exception as e:
            print(f"❌ {config_info['name']} 실패: {e}")
            results[config_info['name']] = None
    
    # ================================
    # 📊 결과 요약
    # ================================
    
    print(f"\\n{'='*60}")
    print("📊 배치 처리 결과 요약")
    print(f"{'='*60}")
    
    for name, result in results.items():
        if result is not None:
            total_experiments = len([exp for exp in result.values() if any(cond is not None for cond in exp.values())])
            print(f"✅ {name}: {total_experiments}개 실험 완료")
        else:
            print(f"❌ {name}: 실패")
    
    return results

# # 12. Fixed Baseline 기능 사용 예시

# # ================================
# # 🎯 Fixed Baseline 사용 예시
# # ================================

def validate_baseline_paths(config_list: List[Dict[str, Any]]) -> None:
    """Baseline 경로 유효성을 검사합니다."""
    print("📋 Baseline 경로 유효성 검사:")
    for config_info in config_list:
        name = config_info["name"]
        if config_info.get("fixed_baseline", False):
            baseline_path = config_info.get("baseline_parent_path")
            if baseline_path and os.path.exists(baseline_path):
                print(f"  ✅ {name}: Baseline 경로 유효 ({baseline_path})")
            else:
                print(f"  ❌ {name}: Baseline 경로 없음 또는 무효 ({baseline_path})")
        else:
            print(f"  📂 {name}: 동일 경로 사용 (fixed_baseline=False)")

# # ================================
# # 🔥 간편한 Fixed Baseline 헬퍼 함수
# # ================================

def create_fixed_baseline_config(
    name: str,
    experiment_path: str,
    baseline_path: str,
    conditions: Dict[str, str],
    baseline_key: str = "baseline"
) -> Dict[str, Any]:
    """Fixed baseline 설정을 쉽게 생성합니다."""
    return {
        "name": name,
        "parent_path": experiment_path,
        "fixed_baseline": True,
        "baseline_parent_path": baseline_path,
        "baseline_condition_key": baseline_key,
        "conditions": conditions
    }