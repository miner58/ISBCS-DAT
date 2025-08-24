import os
import glob
import pandas as pd
from ray import tune
from src.training.hyperparameter_tuning.tune_train.DMMR_train_func import DMMR_train_func, test_func as dmmr_test_func
from src.training.hyperparameter_tuning.runner_raytune import EEGDARayTuneRunner

def make_data_config_path_dict(data_config_base_dir, data_config_path_dict):
    """기존 함수와 동일"""
    for folder_name in data_config_path_dict.keys():
        folder_path = os.path.join(data_config_base_dir, folder_name)
        json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
        data_config_path_dict[folder_name] = json_files
    print(data_config_path_dict)
    return data_config_path_dict


class DMMREEGDARayTuneAnalyzer(EEGDARayTuneRunner):
    """
    DMMR 전용 실험 분석기
    기존 EEGDARayTuneAnalyzer를 상속하여 DMMR 특화 기능 추가
    """
    def __init__(self, config_path, experiment_path, 
                 train_func, test_func, test_data_default_path, data_config_path_dict):
        """
        DMMR 전용 분석기 초기화
        
        Args:
            config_path (str): Ray Tune config 파일 경로
            experiment_path (str): 실험 결과 경로
            train_func (callable): DMMR 훈련 함수
            test_func (callable): DMMR 테스트 함수
            test_data_default_path (str): 테스트 데이터 기본 경로
            data_config_path_dict (dict): 데이터 설정 경로 딕셔너리
        """
        super(DMMREEGDARayTuneAnalyzer, self).__init__(config_path, train_func=train_func)
        self.experiment_path = experiment_path
        self.test_func = test_func
        self.results = None
        self.checkpoint_metric = None
        self.test_data_default_path = test_data_default_path
        self.data_config_path_dict = data_config_path_dict
        self.output_path = os.path.join(os.path.dirname(self.experiment_path), 'analyzing_result')
        os.makedirs(self.output_path, exist_ok=True)

    def set_config_dict(self, config_dict):
        self.config_dict = config_dict

    def load_tuning_results(self):
        """Ray Tune 결과 로드 (DMMR 특화)"""
        self.tunner_setup()
        scheduler_config = self.ray_tune_config['tune_parameters']['scheduler']
        self.checkpoint_metric = scheduler_config['parameters']['metric']

        print(f"Loading DMMR results from {self.experiment_path}...")
        restored_tuner = tune.Tuner.restore(self.experiment_path, self.trainer)
        self.results = restored_tuner.get_results()

    def print_results(self):
        """DMMR 결과 출력"""
        if not self.results:
            print("No DMMR results loaded.")
            return

        for i, result in enumerate(self.results):
            if result.error:
                print(f"DMMR Trial #{i} had an error:", result.error)
                continue
            print(f"DMMR Trial #{i} finished successfully with a mean accuracy metric of:",
                  result.metrics[self.checkpoint_metric])

    def run_test_func(self):
        """DMMR 테스트 함수 실행"""
        if not self.results:
            print("No DMMR results found. Please load the tuning results first.")
            return

        best_result_df = self.results.get_dataframe(filter_metric=self.checkpoint_metric, filter_mode="max")
        best_result_df.to_csv(os.path.join(self.output_path, 'best_result_df.csv'))

        combined_all_df = pd.DataFrame()
        for i, result in enumerate(self.results):
            test_results_df = self._run_test_on_best_checkpoint(result, best_result_df.iloc[i])
            combined_all_df = pd.concat([combined_all_df, test_results_df], ignore_index=True)

        combined_all_df.to_csv(os.path.join(self.output_path, 'raw_test_results.csv'), index=False)
    
    def extract_best_checkpoint(self):
        """DMMR 최적 체크포인트 추출"""
        def validate_checkpoint_path(checkpoint_path):
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"DMMR Checkpoint path does not exist: {checkpoint_path}")
            return checkpoint_path
        
        if not self.results:
            print("No DMMR results found. Please load the tuning results first.")
            return

        best_checkpoints = []
        for i, result in enumerate(self.results):
            best_checkpoint = result.get_best_checkpoint(self.checkpoint_metric, "max")
            validate_checkpoint_path(best_checkpoint.path)
            best_checkpoints.append(best_checkpoint.path)

        # Save the best checkpoints to a CSV file
        best_checkpoints = [{'path': path} for path in best_checkpoints]
        best_checkpoints_df = pd.DataFrame(best_checkpoints)
        best_checkpoints_df.to_csv(os.path.join(self.output_path, 'best_checkpoints.csv'), index=False)

    def _run_test_on_best_checkpoint(self, result, best_row):
        """DMMR 최적 체크포인트에서 테스트 실행"""
        best_checkpoint = result.get_best_checkpoint(self.checkpoint_metric, "max")
        dir_name = os.path.dirname(best_checkpoint.path)

        # Extracting best checkpoint elements
        filtered_elements = list(filter(lambda x: x[0] == best_checkpoint, result.best_checkpoints))[0]
        best_checkpoint = filtered_elements[0]
        best_config = filtered_elements[1]

        # Running tests for each subject configuration
        test_results_list = []
        for subject_name, sub_config_path_list in self.data_config_path_dict.items():
            for sub_config_path in sub_config_path_list:
                test_results = self.test_func(best_config['config']['train_loop_config'], best_checkpoint.path, sub_config_path, self.test_data_default_path)
                test_result = test_results[0]
                test_result['test_subject_name'] = subject_name
                test_result['test_config_path'] = sub_config_path
                test_results_list.append(test_result)

        # Combine the results with the best result row
        test_results_df = pd.DataFrame(test_results_list)
        combined_row = pd.concat([best_row.to_frame().T.reset_index(drop=True)] * len(test_results_df), ignore_index=True)
        combined_row = pd.concat([combined_row, test_results_df.reset_index(drop=True)], axis=1)
        
        return combined_row


def is_dmmr_cross_domain_experiment(experiment_path):
    """
    DMMR Cross-domain 실험인지 판별하는 함수
    wireless2wire 또는 wire2wireless가 경로에 포함되면 True 반환
    """
    return "wireless2wire" in experiment_path or "wire2wireless" in experiment_path


def make_dmmr_cross_domain_data_config_dict(experiment_path, data_config_base_dir):
    """
    DMMR Cross-domain 실험용 테스트 설정 파일을 사용하는 data_config_path_dict 생성
    
    Args:
        experiment_path (str): 실험 경로
        data_config_base_dir (str): 데이터 설정 기본 디렉토리
        
    Returns:
        dict: 테스트용 데이터 설정 경로 딕셔너리
    """
    if "wireless2wire" in experiment_path:
        # raw5&6 → raw3 테스트 (wireless to wire)
        subjects = ["B202", "N201", "R203"]
        test_config_dir = os.path.join(data_config_base_dir.replace("raw5and6config", "raw3config"), "test", "DMMR")
    elif "wire2wireless" in experiment_path:
        # raw3 → raw5&6 테스트 (wire to wireless)
        subjects = ["B112", "N304", "N310"]
        test_config_dir = os.path.join(data_config_base_dir.replace("raw3config", "raw5and6config"), "test", "DMMR")
    else:
        print(f"Warning: Unknown DMMR cross-domain experiment type in {experiment_path}")
        return {"UNKNOWN": []}
    
    data_config_path_dict = {}
    for subject in subjects:
        config_path = os.path.join(test_config_dir, f"dataConfigStim_DMMR_subject{subject}.json")
        if os.path.exists(config_path):
            data_config_path_dict[subject] = [config_path]
        else:
            print(f"Warning: DMMR test config file not found: {config_path}")
            data_config_path_dict[subject] = []
    
    return data_config_path_dict


def extract_max_acc_rows(csv_path, output_path, group_cols=[], metric="test/report/macro avg/accuracy"):
    """
    CSV 파일에서 test_target_acc(또는 test_acc)가 가장 높은 행만 추출하여,
    subject_name이 있으면 (subject_name, test_subject_name) 그룹별로,
    없으면 test_subject_name 그룹별로 최대값을 찾는다.
    결과는 기존 csv_path(원본 파일명)로 덮어쓴다.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'config/train_loop_config/grl_lambda': 'grl_lambda',
        'config/train_loop_config/lnl_lambda': 'lnl_lambda',
        'config/train_loop_config/subject_name': 'subject_name',
        'test_subject_name' : 'test_subject_name',
        'test_acc': 'test_target_acc',
        'test_f1': 'test_target_f1'
    })
    
    # Check if 'test_target_acc' exists
    if metric not in df.columns:
        print(f"{metric} column not found in the data.")
        return
    
    # subject_name 컬럼이 있는지 확인 (NaN만 있는지 여부도 함께 체크할 수도 있음)
    has_subject_name = ('subject_name' in df.columns)
    
    if has_subject_name and not df['subject_name'].isnull().all():
        # subject_name 이 있고, 전부 NaN이 아니라면 (subject_name, test_subject_name)별 최대 acc를 찾는다.
        df = df[df['subject_name'] == df['test_subject_name']]
        group_cols.extend(['subject_name', 'test_subject_name'])
    else:
        # subject_name이 없거나, 전부 NaN이면 기존 로직: test_subject_name별 최대 acc
        group_cols.extend(['test_subject_name'])

    # groupby 후 idxmax를 이용해 test_target_acc가 가장 큰 행의 인덱스를 찾는다.
    max_acc_indices = df.groupby(group_cols)[metric].idxmax()
    
    # 해당 인덱스의 행만 추출
    max_acc_df = df.loc[max_acc_indices]
    
    if 'subject_name' in max_acc_df.columns:
        if 'config/train_loop_config/day_combination' in max_acc_df.columns:
            max_acc_df = max_acc_df.sort_values(by=['subject_name', 'config/train_loop_config/day_combination'])
        else:
            max_acc_df = max_acc_df.sort_values(by=['subject_name'])
    else:
        # subject_name이 없으면 기존 그룹 컬럼 기준 정렬
        max_acc_df = max_acc_df.sort_values(by=group_cols)
    
    # 기존 csv_path 파일명을 그대로 사용 (덮어쓰기)
    output_csv = os.path.join(output_path, f'max_metric_per_subject.csv')

    # 결과를 저장 (덮어쓰기)
    max_acc_df.to_csv(output_csv, index=False)
    
    print(f"DMMR rows with maximum metric per group saved to {output_csv}")


def main_dmmr(experiment_path_list, test_data_default_path, data_config_base_dir):
    """
    DMMR 실험 메인 함수
    
    Args:
        experiment_path_list (list): DMMR 실험 경로 리스트
        test_data_default_path (str): 테스트 데이터 기본 경로
        data_config_base_dir (str): 데이터 설정 기본 디렉토리
    """
    for experiment_path in experiment_path_list:
        print(f"\n=== Processing DMMR experiment: {experiment_path} ===")
        
        # DMMR Cross-domain 실험 여부 확인
        if is_dmmr_cross_domain_experiment(experiment_path):
            print("🔄 DMMR Cross-domain experiment detected")
            data_config_path_dict = make_dmmr_cross_domain_data_config_dict(experiment_path, data_config_base_dir)
        else:
            print("🏠 DMMR Inner experiment detected")
            if "raw3" in experiment_path:
                subjects = ["B202", "N201", "R203"]
                test_config_dir = os.path.join(data_config_base_dir, "raw3config", "test", "DMMR")
            else:  # raw5&6
                subjects = ["B112", "N304", "N310"]
                test_config_dir = os.path.join(data_config_base_dir, "raw5and6config", "test", "DMMR")
            
            # Individual subject test configs
            data_config_path_dict = {}
            for subject in subjects:
                config_path = os.path.join(test_config_dir, f"dataConfigStim_DMMR_subject{subject}.json")
                if os.path.exists(config_path):
                    data_config_path_dict[subject] = [config_path]
                else:
                    print(f"Warning: DMMR test config not found: {config_path}")
                    data_config_path_dict[subject] = []
        
        # Find config file
        config_files = glob.glob(os.path.join(experiment_path, "*.yml"))
        if not config_files:
            print(f"No .yml file found in the directory: {experiment_path}")
            continue
        config_path = config_files[0]
        
        # Determine experiment name for result directory
        if "wireless2wire" in experiment_path:
            exp_name = 'dmmr_tune_wireless2wire'
        elif "wire2wireless" in experiment_path:
            exp_name = 'dmmr_tune_wire2wireless'
        elif "raw3" in experiment_path:
            exp_name = 'dmmr_tune_raw3'
        elif "raw5" in experiment_path:
            exp_name = 'dmmr_tune_raw5and6'
        else:
            exp_name = 'dmmr_tune'
        
        # Initialize DMMR analyzer
        analyzer = DMMREEGDARayTuneAnalyzer(
            config_path, 
            os.path.join(experiment_path, exp_name), 
            DMMR_train_func, 
            dmmr_test_func, 
            test_data_default_path, 
            data_config_path_dict
        )
        
        # Load results and analyze them
        analyzer.load_tuning_results()
        analyzer.print_results()
        analyzer.run_test_func()
        
        # Extract maximum accuracy results
        extract_max_acc_rows(
            os.path.join(experiment_path, 'analyzing_result', 'raw_test_results.csv'),
            os.path.join(experiment_path, 'analyzing_result'), 
            group_cols=[], 
            metric="test_micro_acc"
        )


def extract_best_checkpoint_for_each_dmmr_experiment(experiment_path_list, data_config_base_dir=None):
    """
    각 DMMR 실험의 최적 체크포인트를 추출하고 CSV 파일로 저장
    """
    print("🔍 Extracting best checkpoints for each DMMR experiment...")
    
    for experiment_path in experiment_path_list:
        # Cross-domain 실험인지 확인하고 적절한 설정 사용
        if is_dmmr_cross_domain_experiment(experiment_path):
            print(f"DMMR Cross-domain experiment detected: {experiment_path}")
            data_config_path_dict = make_dmmr_cross_domain_data_config_dict(experiment_path, data_config_base_dir)
        else:
            # Inner 실험용 더미 dict
            data_config_path_dict = {}
        
        config_files = glob.glob(os.path.join(experiment_path, "*.yml"))
        if not config_files:
            continue
        config_path = config_files[0]
        
        # Determine experiment name
        if "wireless2wire" in experiment_path:
            exp_name = 'dmmr_tune_wireless2wire'
        elif "wire2wireless" in experiment_path:
            exp_name = 'dmmr_tune_wire2wireless'
        elif "raw3" in experiment_path:
            exp_name = 'dmmr_tune_raw3'
        elif "raw5" in experiment_path:
            exp_name = 'dmmr_tune_raw5and6'
        else:
            exp_name = 'dmmr_tune'
        
        analyzer = DMMREEGDARayTuneAnalyzer(
            config_path, 
            os.path.join(experiment_path, exp_name), 
            DMMR_train_func, 
            dmmr_test_func, 
            test_data_default_path, 
            data_config_path_dict
        )
        analyzer.load_tuning_results()
        analyzer.extract_best_checkpoint()
    
    print("✅ Best checkpoints extracted for all DMMR experiments.")


def filter_dmmr_experiment_path_list(experiment_path_list, experiment_type):
    """
    DMMR 실험 경로 리스트를 필터링
    
    Args:
        experiment_path_list (list): 실험 경로 리스트
        experiment_type (str): "inner_wire", "inner_wireless", "wireless2wire", "wire2wireless"
    
    Returns:
        list: 필터링된 실험 경로 리스트
    """
    if experiment_type == "inner_wire":
        return [path for path in experiment_path_list if ("raw3" in path and "DMMR" in path and 
                                                         "wireless2wire" not in path and "wire2wireless" not in path)]
    elif experiment_type == "inner_wireless":
        return [path for path in experiment_path_list if ("raw5" in path and "DMMR" in path and 
                                                         "wireless2wire" not in path and "wire2wireless" not in path)]
    elif experiment_type == "wireless2wire":
        return [path for path in experiment_path_list if "wireless2wire" in path]
    elif experiment_type == "wire2wireless":
        return [path for path in experiment_path_list if "wire2wireless" in path]
    else:
        raise ValueError(f"Invalid DMMR experiment_type: {experiment_type}")


def get_sub_folder_list(folder_list):
    """일반화된 하위 폴더 리스트 생성 함수"""
    sub_folder_list = []
    for folder in folder_list:
        print(f">> folder: {folder}")
        if os.path.exists(folder):
            tmp_dir = os.listdir(folder)
            print(f"tmp_dir:\n {tmp_dir}")
            sub_folder_list.extend([os.path.join(folder, folder_name) for folder_name in tmp_dir])
        else:
            print(f"Warning: Folder does not exist: {folder}")
    return sub_folder_list


if __name__ == "__main__":
    # DMMR 실험 설정
    test_data_default_path = "/root/workspace/Fairness_for_generalization"
    data_config_base_dir = "/root/workspace/Fairness_for_generalization/src/config/data_config"
    
    # DMMR 실험 경로 예시 (실제 실험 결과 경로로 수정 필요)
    tmp_folder_list = [
        # DMMR inner experiments
        "/root/workspace/Fairness_for_generalization/results_DMMR_ECoG"
        # "/root/workspace/Fairness_for_generalization/results_DMMR_ECoG/DMMR_1_inner_wireless",
        # "/root/workspace/Fairness_for_generalization/results_trans/DMMR_experiments/DMMR_2_inner_wire",
        # DMMR cross-domain experiments  
        # "/root/workspace/Fairness_for_generalization/results_trans/DMMR_experiments/DMMR_3_wireless2wire",
        # "/root/workspace/Fairness_for_generalization/results_trans/DMMR_experiments/DMMR_4_wire2wireless",
    ]
    
    experiment_path_list = []
    if tmp_folder_list:
        experiment_path_list = get_sub_folder_list(tmp_folder_list)
        
        # DMMR 실험 타입별 필터링 (예시)
        # experiment_path_list = filter_dmmr_experiment_path_list(experiment_path_list, "wireless2wire")
        
        # 하위 폴더 확장
        experiment_path_list = get_sub_folder_list(experiment_path_list)
        
        # Ray 결과만 필터링
        experiment_path_list = [path for path in experiment_path_list if "ray_results" in path]

        print(experiment_path_list)
        
        # 최적 체크포인트 추출
        extract_best_checkpoint_for_each_dmmr_experiment(experiment_path_list, data_config_base_dir)

        
        # 메인 분석 실행
        main_dmmr(experiment_path_list, test_data_default_path, data_config_base_dir)
    else:
        print("No DMMR experiment folders specified. Please update tmp_folder_list with actual experiment paths.")