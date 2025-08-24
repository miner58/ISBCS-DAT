import os
import glob
import pandas as pd
from ray import tune
from src.training.hyperparameter_tuning.tune_train.base_train_func import test_func, train_func
from src.training.hyperparameter_tuning.tune_train.groupDRO_train_func import test_func as groupDRO_test_func, train_func as groupDRO_train_func
from src.training.hyperparameter_tuning.runner_raytune import EEGDARayTuneRunner

def is_cross_domain_experiment(experiment_path):
    """
    experiment_path에서 '3_wireless2wire' 또는 '4_wire2wireless'가 포함되어 있는지 확인합니다.
    """
    return '3_wireless2wire' in experiment_path or '4_wire2wireless' in experiment_path

def make_cross_domain_data_config_dict(data_config_base_dir, data_type):
    """
    Cross-domain 실험용 ALL 설정 파일을 사용하는 data_config_path_dict를 생성합니다.
    """
    cross_domain_config_dict = {}
    all_folder_path = os.path.join(data_config_base_dir, "ALL")
    
    if os.path.exists(all_folder_path):
        json_files = glob.glob(os.path.join(all_folder_path, '**', '*.json'), recursive=True)
        cross_domain_config_dict["ALL"] = json_files
        print(f"Cross-domain config dict for {data_type}: {cross_domain_config_dict}")
    else:
        print(f"Warning: ALL config folder not found at {all_folder_path}")
        cross_domain_config_dict["ALL"] = []
    
    return cross_domain_config_dict

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

def main(experiment_path_list, test_data_default_path, data_config_path_dict, train_func, test_func, data_config_base_dir=None, data_type=None):
    # Cross-domain 실험인지 확인하고 적절한 설정 사용
    if data_config_base_dir and data_type and any(is_cross_domain_experiment(exp_path) for exp_path in experiment_path_list):
        print(f"Cross-domain experiment detected for {data_type}. Using ALL configuration files.")
        data_config_path_dict = make_cross_domain_data_config_dict(data_config_base_dir, data_type)
    else:
        # 기존 로직: 개별 그룹별 설정 사용
        data_config_path_dict = make_data_config_path_dict(data_config_base_dir or "", data_config_path_dict)

    for experiment_path in experiment_path_list:
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

def extract_best_checkpoint_for_each_experiment(experiment_path_list, data_config_base_dir=None, data_type=None):
    """
    Extracts the best checkpoint for each experiment and saves it to a CSV file.
    """
    # print lightning emoji
    print("🔍 Extracting best checkpoints for each experiment...")
    # Cross-domain 실험인지 확인하고 적절한 설정 사용
    if data_config_base_dir and data_type and any(is_cross_domain_experiment(exp_path) for exp_path in experiment_path_list):
        print(f"Cross-domain experiment detected for {data_type}. Using ALL configuration files.")
        data_config_path_dict = make_cross_domain_data_config_dict(data_config_base_dir, data_type)
    else:
        # 기본 설정이 필요한 경우를 위한 더미 dict
        data_config_path_dict = {}
    
    for experiment_path in experiment_path_list:
        config_path = glob.glob(os.path.join(experiment_path, "*.yml"))[0]
        analyzer = EEGDARayTuneAnalyzer(config_path, os.path.join(experiment_path, 'eeg_tune'), 
                                        train_func, test_func, test_data_default_path, data_config_path_dict)
        analyzer.load_tuning_results()
        analyzer.extract_best_checkpoint()
    print("✅ Best checkpoints extracted for all experiments.")

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

if __name__ == "__main__":
    # # wire
    # data_type = "wire"
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw3config/test/only1Day1,8"
    # data_config_path_dict = {"B202": [],
    #                         "N201": [],
    #                         "R203": []}

    # wireless
    data_type = "wireless"
    test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/raw5&6config/test/only1Day1,8"
    data_config_path_dict={"B112":[],
                      "N304":[],
                      "N310":[]}
    
    # # UI
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/UIconfig/test/group"
    # data_config_path_dict={"A":[],
    #                   "B":[],
    #                   "C":[]}

    # UNM
    # test_data_default_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization"
    # data_config_base_dir = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/UNMconfig/test/group"
    # data_config_path_dict={"A":[],
    #                   "B":[],
    #                   "C":[]}

    tmp_folder_list = [
        # # lnl_lag_ASR_O
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/lnl_lag_ASR_O/LNLlag_vs_LNL",
        # # results_new_exp/results_tmp2_full/base_vs_learnable
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_new_exp/results_tmp2_full/base_vs_learnable",
        # # results_new_exp/results_tmp2_full/base_vs_LNL
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_new_exp/results_tmp2_full/base_vs_LNL",
        # # results_trans/results_da_one
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_da_one",
        # # results_trans/results_da_one_60
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_da_one_60",
        # # results_trans/results_subject_20
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_20",
        # # results_trans/results_subject_60
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60",
        # results_subject_20_EEGNet_ASR_O
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_20_EEGNet_ASR_O",
        # results_subject_60_EEGNet_ASR_O
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60_EEGNet_ASR_O",
        # channel norm
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_channel_norm_ASR",

        # results subject 20 EEGNet ASR O
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_20_EEGNet_ASR_O",
        # results subject 60 EEGNet ASR O
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60_EEGNet_ASR_O",

        # #groupDRO
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_groupDRO_ASR",
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_groupDRO_ASR_NEW",

        # subject softlabel 60
        # "/home/jsw/Fairness/tmp/Fairness_for_generalization/results_trans/results_trans/results_subject_60_LNLsoftlabel_ASR_O"
        "/root/workspace/Fairness_for_generalization/results_trans/results_trans/results_subject_60"
    ]

    experiment_path_list = [
    ]
    experiment_path_list = get_sub_folder_list(tmp_folder_list)

    experiment_path_list = filter_experiment_path_list(experiment_path_list, data_type)

    # print lightning emoji
    print("\U0001F6A8")

    experiment_path_list = get_sub_folder_list(experiment_path_list)

    experiment_path_list = filter_experiment_path_list_ray_results(experiment_path_list)

    extract_best_checkpoint_for_each_experiment(experiment_path_list, data_config_base_dir, data_type)
    # main(experiment_path_list, test_data_default_path, data_config_path_dict, train_func=train_func, test_func=test_func, data_config_base_dir=data_config_base_dir, data_type=data_type)
    # main(experiment_path_list, test_data_default_path, data_config_path_dict, train_func=groupDRO_train_func, test_func=groupDRO_test_func, data_config_base_dir=data_config_base_dir, data_type=data_type)
