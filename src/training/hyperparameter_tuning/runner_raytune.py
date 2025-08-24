import os
import sys

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

# 상대 경로로 내부 모듈 import
from src.training.hyperparameter_tuning.base_runner_raytune import BaseTuneRunner
from src.training.hyperparameter_tuning.tune_train.base_train_func import train_func
from src.training.hyperparameter_tuning.tune_train.groupDRO_train_func import train_func as groupDRO_train_func

# DMMR train function (with error handling)
try:
    from src.training.hyperparameter_tuning.tune_train.DMMR_train_func import DMMR_train_func
    print("DMMR train function imported successfully.")
    _DMMR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DMMR train function not available: {e}")
    DMMR_train_func = None
    _DMMR_AVAILABLE = False

class EEGDARayTuneRunner(BaseTuneRunner):
    def __init__(self, config_path, **kwargs):
        super(EEGDARayTuneRunner, self).__init__(config_path)
        # 실행 과정에서 참조할 필드
        self.ray_tune_config = None
        self.search_space = None
        self.scheduler = None
        self.reporter = None
        self.scaling_config = None
        self.run_config = None
        self.tune_config = None
        self.trainer = None
        self.results = None
        self.best_result = None
        self.train_func = kwargs['train_func']


    def create_scheduler(self):
        """스케줄러(예: ASHAScheduler) 생성"""
        scheduler_config = self.ray_tune_config['tune_parameters']['scheduler']
        if scheduler_config['name'] == "ASHAScheduler":
            self.scheduler = ASHAScheduler(
                metric=scheduler_config['parameters']['metric'],
                mode=scheduler_config['parameters']['mode'],
                max_t=scheduler_config['parameters']['max_t'],
                grace_period=scheduler_config['parameters']['grace_period'],
                reduction_factor=scheduler_config['parameters']['reduction_factor']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")

    def create_reporter(self):
        """Reporter(예: CLIReporter) 생성"""
        reporter_config = self.ray_tune_config['tune_parameters']['reporter']
        if reporter_config['name'] == "CLIReporter":
            self.reporter = CLIReporter(
                metric_columns=reporter_config['parameters']['metric_columns']
            )
        else:
            raise ValueError(f"Unsupported reporter: {reporter_config['name']}")

    def create_scaling_config(self):
        """리소스 사용량 지정"""
        tune_params = self.ray_tune_config['tune_parameters']
        self.scaling_config = ScalingConfig(
            num_workers=tune_params['num_workers'],
            use_gpu=True,
            resources_per_worker=tune_params['resources_per_worker'],
        )

    def create_run_config(self):
        """실행 설정 지정"""
        tune_params = self.ray_tune_config['tune_parameters']
        self.run_config = RunConfig(
            name=tune_params['run']['name'],
            storage_path=tune_params['run']['storage_path'],
            log_to_file=tune_params['run']['log_to_file'],
            verbose=tune_params['run']['verbose'],
            progress_reporter=self.reporter,
            checkpoint_config=CheckpointConfig(
                num_to_keep=tune_params['checkpoint']['num_to_keep'],
                checkpoint_score_attribute=tune_params['checkpoint']['checkpoint_score_attribute'],
                checkpoint_score_order=tune_params['checkpoint']['checkpoint_score_order'],
            ),
        )

    def create_tune_config(self):
        """TuneConfig 생성"""
        tune_params = self.ray_tune_config['tune_parameters']
        self.tune_config = tune.TuneConfig(
            num_samples=tune_params['num_samples'],
            max_concurrent_trials=tune_params['max_concurrent_trials'],
            scheduler=self.scheduler,
        )

    def create_trainer(self):
        """Ray의 TorchTrainer 생성"""
        self.trainer = TorchTrainer(
            train_loop_per_worker=self.train_func,
            scaling_config=self.scaling_config,
            run_config=self.run_config,
        )

    def run_tuner(self):
        """Ray Tune을 실행하고 결과를 저장"""
        param_space = {"train_loop_config": self.search_space}
        tuner = tune.Tuner(
            self.trainer,
            param_space=param_space,
            tune_config=self.tune_config
        )
        self.results = tuner.fit()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 여러 개의 config_path에 대해 순차 실행할 수 있음
    config_path_list = [
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/src/config/raytune_config/RayTune/raw3/lr_schedular_ReduceLROnPlateau_allMouse_groupDRO.yml",
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/src/config/raytune_config/RayTune/raw3/lr_schedular_ReduceLROnPlateau_test1_groupDRO.yml",
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/src/config/raytune_config/RayTune/raw5&6/lr_schedular_ReduceLROnPlateau_test1_groupDRO.yml",
        "/home/jsw/Fairness/tmp/Fairness_for_generalization/src/config/raytune_config/RayTune/raw5&6/lr_schedular_ReduceLROnPlateau_allMouse_groupDRO.yml",
    ]

    for config_path in config_path_list:
        print(f"Running tuning for config: {config_path}")
        runner = EEGDARayTuneRunner(config_path, train_func=groupDRO_train_func)
        runner.run()  # 모든 단계 일괄 실행


def main_dmmr():
    """Run DMMR experiments."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 for DMMR
    
    if not _DMMR_AVAILABLE:
        print("❌ DMMR train function not available. Check dependencies.")
        return
    
    # DMMR config paths (using updated paths)
    dmmr_config_list = [
        # "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/raw3/DMMR_test.yml",
        # "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/raw5&6/DMMR_test.yml",
        # "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/raw3/DMMR_allmouse.yml",
        # "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/raw5&6/DMMR_allmouse.yml"
        # "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/UI/DMMR_allMouse.yml",
        "/root/workspace/Fairness_for_generalization/src/config/raytune_config/RayTune/UNM/DMMR_allMouse.yml",
    ]
    
    for config_path in dmmr_config_list:
        print(f"🧠 Running DMMR tuning for config: {config_path}")
        try:
            runner = EEGDARayTuneRunner(config_path, train_func=DMMR_train_func)
            runner.run()
            print(f"✅ DMMR experiment completed: {config_path}")
        except Exception as e:
            print(f"❌ DMMR experiment failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # 메모리 모니터링 완전 비활성화
    os.environ["RAY_memory_usage_threshold"] = "1.0"   # 100%까지 사용 허용

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dmmr":
        main_dmmr()
    else:
        main()