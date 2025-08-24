import os
import glob
import pandas as pd
from ray import tune
from tune_train.base_train_func import test_func
from tune_train.groupDRO_train_func import test_func as groupDRO_test_func
from tune_train.base_train_func import train_func
from tune_train.groupDRO_train_func import train_func as groupDRO_train_func
from runner_raytune import EEGDARayTuneRunner

def make_data_config_path_dict(data_config_base_dir, data_config_path_dict):
    for folder_name in data_config_path_dict.keys():
        folder_path = os.path.join(data_config_base_dir, folder_name)
        json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
        data_config_path_dict[folder_name] = json_files
    print(data_config_path_dict)

    return data_config_path_dict


class EEGDARayTuneAnalyzer(EEGDARayTuneRunner):
    def __init__(self, config_path, experiment_path, 
                 train_func, test_func, test_data_default_path, data_config_path_dict):
        """
        Initializes the EEGDARayTuneAnalyzer class.

        Args:
            experiment_path (str): The path to the Ray Tune experiment.
            test_func (callable): The function to test the model on the test data.
        """
        super(EEGDARayTuneAnalyzer, self).__init__(config_path, train_func=train_func)
        self.experiment_path = experiment_path
        self.test_func = test_func
        self.results = None
        self.checkpoint_metric = None
        self.test_data_default_path = test_data_default_path
        self.data_config_path_dict=data_config_path_dict
        self.output_path = os.path.join( os.path.dirname(self.experiment_path), 'analyzing_result')
        os.makedirs(self.output_path, exist_ok=True)

    def set_config_dict(self, config_dict):
        self.config_dict = config_dict


    def load_tuning_results(self):
        """
        Loads the tuning results from the experiment path.
        """
        self.tunner_setup()
        scheduler_config = self.ray_tune_config['tune_parameters']['scheduler']
        self.checkpoint_metric = scheduler_config['parameters']['metric']

        print(f"Loading results from {self.experiment_path}...")
        restored_tuner = tune.Tuner.restore(self.experiment_path, self.trainer)
        self.results = restored_tuner.get_results()

    def print_results(self):
        """
        Prints the results of the tuning process.
        """
        if not self.results:
            print("No results loaded.")
            return

        for i, result in enumerate(self.results):
            if result.error:
                print(f"Trial #{i} had an error:", result.error)
                continue
            print(f"Trial #{i} finished successfully with a mean accuracy metric of:",
                  result.metrics[self.checkpoint_metric])

    def run_test_func(self):
        """
        Runs the test function on the best checkpoints of each trial.
        """
        if not self.results:
            print("No results found. Please load the tuning results first.")
            return

        best_result_df = self.results.get_dataframe(filter_metric=self.checkpoint_metric, filter_mode="max")
        best_result_df.to_csv(os.path.join(self.output_path, 'best_result_df.csv'))

        combined_all_df = pd.DataFrame()
        for i, result in enumerate(self.results):
            test_results_df = self._run_test_on_best_checkpoint(result, best_result_df.iloc[i])
            combined_all_df = pd.concat([combined_all_df, test_results_df], ignore_index=True)

        combined_all_df.to_csv(os.path.join(self.output_path, 'raw_test_results.csv'), index=False)
    
    def extract_best_checkpoint(self):
        """
        Extracts the best checkpoint from each trial's results.
        """
        def validate_checkpoint_path(checkpoint_path):
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
            return checkpoint_path
        if not self.results:
            print("No results found. Please load the tuning results first.")
            return

        best_checkpoints = []
        for i, result in enumerate(self.results):
            best_checkpoint = result.get_best_checkpoint(self.checkpoint_metric, "max")
            validate_checkpoint_path(best_checkpoint.path)
            best_checkpoints.append(best_checkpoint.path)

        # Save the best checkpoints to a CSV file
        # column: ['path']
        best_checkpoints = [{'path': path} for path in best_checkpoints]
        best_checkpoints_df = pd.DataFrame(best_checkpoints)
        best_checkpoints_df.to_csv(os.path.join(self.output_path, 'best_checkpoints.csv'), index=False)


    def _run_test_on_best_checkpoint(self, result, best_row):
        """
        Runs the test function on the best checkpoint of a single trial.

        Args:
            result: The result of a single trial.
            best_row: The corresponding row from the best result dataframe.

        Returns:
            DataFrame containing test results for the current trial.
        """
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
        # for _, test_res in test_results_df.iterrows():
        #     combined_row = pd.concat([best_row.to_frame().T.reset_index(drop=True), test_res.to_frame().T.reset_index(drop=True)], axis=1)
        return combined_row

    def analyze_results(self):
        """
        Analyzes the results of the tuning, typically by selecting the best configuration
        and testing on the test data.
        """
        if not self.results:
            print("No results found. Please load the tuning results first.")
            return

        # Extract the best result
        best_result = self.results.get_best_result(metric=self.checkpoint_metric, mode="max")
        print(f"Best result: {best_result}")

        # Perform the test with the best configuration on the test data
        best_checkpoint = best_result.checkpoint
        test_data = self.test_func(best_checkpoint)
        print("Test Results:", test_data)

def extract_max_acc_rows(csv_path, output_path, group_cols=[], metric="test/report/macro avg/accuracy"):
    """
    CSV 파일에서 test_target_acc(또는 test_acc)가 가장 높은 행만 추출하여,
    subject_name이 있으면 (subject_name, test_subject_name) 그룹별로,
    없으면 test_subject_name 그룹별로 최대값을 찾는다.
    결과는 기존 csv_path(원본 파일명)로 덮어쓴다.
    """

    # metric = "test/report/macro avg/accuracy"
    # metric = "test_target_acc"

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

    # subject_name 컬럼 값이 A, B, C 등으로 되어 있으면 groupA, groupB, groupC로 변경
    if 'subject_name' in df.columns:
        df['subject_name'] = df['subject_name'].replace({'A': 'groupA', 'B': 'groupB', 'C': 'groupC'})
    
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
    # 출력 경로만 다르게 지정하고 싶다면 아래에서 csv_path 대신 다른 경로를 쓰면 됨
    output_csv = os.path.join(output_path, f'max_metric_per_subject.csv')

    # 결과를 저장 (덮어쓰기)
    max_acc_df.to_csv(output_csv, index=False)
    
    print(f"Rows with maximum metric per group saved (overwritten) to {output_csv}")

def extract_max_acc_rows_average(csv_path, output_path, metric="test/report/macro avg/accuracy"):
    """
    CSV 파일에서 groupA, groupB, groupC의 평균 성능이 가장 높은 설정을 추출한다.
    각 설정(configuration)에 대해 세 그룹의 평균 성능을 계산하고, 가장 높은 평균을 가진 설정의 모든 행을 추출한다.
    """
    
    df = pd.read_csv(csv_path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'config/train_loop_config/grl_lambda': 'grl_lambda',
        'config/train_loop_config/lnl_lambda': 'lnl_lambda',
        'config/train_loop_config/lr': 'lr',
        'config/train_loop_config/subject_name': 'subject_name',
        'test_subject_name': 'test_subject_name',
        'test_acc': 'test_target_acc',
        'test_f1': 'test_target_f1'
    })
    
    if metric not in df.columns:
        print(f"{metric} column not found in the data.")
        return
    
    # subject_name 컬럼 값 정규화
    if 'subject_name' in df.columns:
        df['subject_name'] = df['subject_name'].replace({'A': 'groupA', 'B': 'groupB', 'C': 'groupC'})
    
    # 설정 식별을 위한 컬럼들 lr, grl_lambda, lnl_lambda
    config_cols = [col for col in df.columns if col in ['grl_lambda', 'lnl_lambda', 'lr']]
    if 'subject_name' in df.columns:
        config_cols.append('subject_name')
    
    # config/train_loop_config/lr 컬럼도 추가 (rename되지 않은 경우)
    if 'config/train_loop_config/lr' in df.columns and 'config/train_loop_config/lr' not in config_cols:
        config_cols.append('config/train_loop_config/lr')
    
    # Debug: 설정 정보 출력
    print(f"Config columns found: {config_cols}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # 빈 config_cols 처리
    if not config_cols:
        print("No configuration columns found. Using trial_id or index as grouping.")
        # trial_id나 다른 식별자가 있는지 확인
        if 'trial_id' in df.columns:
            config_cols = ['trial_id']
        else:
            # 고유한 행을 식별하기 위한 인덱스 생성
            df['config_index'] = df.index
            config_cols = ['config_index']
    
    # 각 설정별로 그룹 평균 계산
    avg_performance = df.groupby(config_cols)[metric].mean().reset_index()
    avg_performance.columns = config_cols + ['avg_metric']
    
    # Debug: 평균 성능 정보 출력
    print(f"Average performance shape: {avg_performance.shape}")
    print(f"Average performance head:\n{avg_performance.head()}")
    
    # 빈 결과 처리
    if avg_performance.empty or avg_performance['avg_metric'].isna().all():
        print("No valid configurations found or all metrics are NaN.")
        return
    
    # 가장 높은 평균 성능을 가진 설정 찾기
    best_config_idx = avg_performance['avg_metric'].idxmax()
    best_config = avg_performance.iloc[best_config_idx]
    
    # 해당 설정의 모든 행 추출
    mask = pd.Series([True] * len(df))
    for col in config_cols:
        if col in df.columns:
            mask &= (df[col] == best_config[col])
    
    best_avg_df = df[mask].copy()
    best_avg_df = best_avg_df.sort_values(by=['test_subject_name'])
    
    # 평균 성능 정보 추가
    best_avg_df['average_metric'] = best_config['avg_metric']
    
    output_csv = os.path.join(output_path, 'max_average_metric_config.csv')
    best_avg_df.to_csv(output_csv, index=False)
    
    print(f"Configuration with highest average metric ({best_config['avg_metric']:.4f}) saved to {output_csv}")
    return best_avg_df

def main(experiment_path_list, test_data_default_path, data_config_base_dir, data_type, train_func, test_func):
    for experiment_path in experiment_path_list:
        print(f"\n=== Processing experiment: {experiment_path} ===")
        
        # Cross-domain 실험 여부 확인
        if is_cross_domain_experiment(experiment_path):
            print("🔄 Cross-domain experiment detected - using ALL config")
            data_config_path_dict = make_cross_domain_data_config_dict(data_type, data_config_base_dir)
        else:
            print("🏠 Inner experiment detected - using individual group configs")
            data_config_path_dict = {"groupA": [], "groupB": [], "groupC": []}
            data_config_path_dict = make_data_config_path_dict(data_config_base_dir, data_config_path_dict)
        
        # Find the .yml file in the experiment_path directory
        config_path = glob.glob(os.path.join(experiment_path, "*.yml"))[0]
        if not config_path:
            raise FileNotFoundError(f"No .yml file found in the directory: {experiment_path}")

        # Initialize the analyzer
        analyzer = EEGDARayTuneAnalyzer(config_path, os.path.join(experiment_path, 'eeg_tune'), 
                                        train_func, test_func, test_data_default_path, data_config_path_dict)

        # Load results and analyze them
        analyzer.load_tuning_results()
        analyzer.print_results()
        analyzer.run_test_func()

        extract_max_acc_rows(os.path.join(experiment_path, 'analyzing_result','raw_test_results.csv'),
                os.path.join(experiment_path, 'analyzing_result'), group_cols=[], metric="test/report/macro avg/accuracy")
        
        # Extract configuration with highest average performance across groups
        extract_max_acc_rows_average(os.path.join(experiment_path, 'analyzing_result','raw_test_results.csv'),
                os.path.join(experiment_path, 'analyzing_result'), metric="test/report/macro avg/accuracy")

def extract_best_checkpoint_for_each_experiment(experiment_path_list):
    """
    Extracts the best checkpoint for each experiment and saves it to a CSV file.
    """
    for experiment_path in experiment_path_list:
        config_path = glob.glob(os.path.join(experiment_path, "*.yml"))[0]
        analyzer = EEGDARayTuneAnalyzer(config_path, os.path.join(experiment_path, 'eeg_tune'), 
                                        train_func, test_func, test_data_default_path, data_config_path_dict)
        analyzer.load_tuning_results()
        analyzer.extract_best_checkpoint()

"""
filter_experiment_path_list 함수는 
data_type에 따라 실험 경로 리스트를 필터링한다.
data_type = "wire" 일 때, 
    - 2_innrer_wire, 3_wireless2wire 경로만 필터링
data_type = "wireless" 일 때, 
    - 1_inner_wireless, 4_wire2wireless 경로만 필터링
"""
def filter_experiment_path_list(experiment_path_list, data_type):
    if data_type == "wire":
        return [path for path in experiment_path_list if "2_inner_wire" in path or "3_wireless2wire" in path]
    elif data_type == "wireless":
        return [path for path in experiment_path_list if "1_inner_wireless" in path or "4_wire2wireless" in path]
    if data_type == "UI":
        return [path for path in experiment_path_list if "1_inner_UI" in path or "4_UNM2UI" in path]
    elif data_type == "UNM":
        return [path for path in experiment_path_list if "2_inner_UNM" in path or "3_UI2UNM" in path]
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

def filter_experiment_path_list_ray_results(experiment_path_list):
    """
    ray_results 라는 단어가 포함된 경로만 필터링한다.
    """
    return [path for path in experiment_path_list if "ray_results" in path]

"""
for tmp_folder in tmp_folder_list:
        # tmp_folder 내부의 폴더 명을 추가
        tmp_dir = os.listdir(tmp_folder)
        experiment_path_list.extend([os.path.join(tmp_folder, folder) for folder in tmp_dir])
위의 코드를 일반화 시키고 함수화 하였음
"""
def get_sub_folder_list(folder_list):
    sub_folder_list = []
    for folder in folder_list:
        print(f">> folder: {folder}")
        tmp_dir = os.listdir(folder)
        print(f"tmp_dir:\n {tmp_dir}")
        sub_folder_list.extend([os.path.join(folder, folder_name) for folder_name in tmp_dir])
    return sub_folder_list

def is_cross_domain_experiment(experiment_path):
    """
    Cross-domain 실험인지 판별하는 함수
    3_UI2UNM 또는 4_UNM2UI가 경로에 포함되면 True 반환
    """
    return "3_UI2UNM" in experiment_path or "4_UNM2UI" in experiment_path

def make_cross_domain_data_config_dict(data_type, data_config_base_dir):
    """
    Cross-domain 실험용 ALL 설정 파일을 사용하는 data_config_path_dict 생성
    """
    if data_type == "UI":
        all_config_path = os.path.join(data_config_base_dir, "ALL", "dataConfigStimGRL_ALL.json")
        if os.path.exists(all_config_path):
            return {"ALL": [all_config_path]}
        else:
            print(f"Warning: ALL config file not found at {all_config_path}")
            return {"ALL": []}
    elif data_type == "UNM":
        # UNM 파일에서 UI 데이터를 테스트하는 경우 (4_UNM2UI)
        ui_config_base_dir = data_config_base_dir.replace("UNMconfig", "UIconfig")
        all_config_path = os.path.join(ui_config_base_dir, "ALL", "dataConfigStimGRL_ALL.json")
        if os.path.exists(all_config_path):
            return {"ALL": [all_config_path]}
        else:
            print(f"Warning: UI ALL config file not found at {all_config_path}")
            return {"ALL": []}
    return {"ALL": []}

if __name__ == "__main__":
    # # wire
    # data_type = "wire"
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw3config/test/only1Day1,8"
    # data_config_path_dict = {"B202": [],
    #                         "N201": [],
    #                         "R203": []}

    # # wireless
    # data_type = "wireless"
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw5&6config/test/only1Day1,8"
    # data_config_path_dict={"B112":[],
    #                   "N304":[],
    #                   "N310":[]}
    
    # UI
    data_type = "UI"
    test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/UIconfig/test"
    data_config_path_dict={"groupA":[],
                      "groupB":[],
                      "groupC":[]}

    # # UNM
    # data_type = "UNM"
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/UNMconfig/test"
    # data_config_path_dict={"groupA":[],
    #                   "groupB":[],
    #                   "groupC":[]}

    tmp_folder_list = [
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_public_20",
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_public_sub_60"
    ]

    experiment_path_list = [
    ]
    experiment_path_list = get_sub_folder_list(tmp_folder_list)

    experiment_path_list = filter_experiment_path_list(experiment_path_list, data_type)

    experiment_path_list = get_sub_folder_list(experiment_path_list)

    experiment_path_list = filter_experiment_path_list_ray_results(experiment_path_list)

    experiment_path_list = [
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_public_20/4_UNM2UI/ray_results_allMouse_UNM_finetune_ReduceLROnPlateau_batch16",
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_public_sub_60/4_UNM2UI/ray_results_allMouse_UNM_finetune_ReduceLROnPlateau_LNL_batch16"
    ]

    # extract_best_checkpoint_for_each_experiment(experiment_path_list)
    main(experiment_path_list, test_data_default_path, data_config_base_dir, data_type, train_func=train_func, test_func=test_func)
    # main(experiment_path_list, test_data_default_path, data_config_base_dir, data_type, train_func=groupDRO_train_func, test_func=groupDRO_test_func)
