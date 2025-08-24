import os
import sys
import random
import json

import numpy as np
import torch

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

# Ray + Lightning 관련
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)

# 사용자 정의 모듈
from src.training.hyperparameter_tuning.tune_train.base_train_func import LightningEEGDARunner
from src.models.eegnetDRO import EEGNetDRO
from src.data.modules.EEGdataModuelGroupDRO import EEGDataModuleGroupDRO


def build_model_args(config, chans, samples, n_groups=None, group_counts=None):
    """
    주어진 config, chans, samples를 기반으로 model_class에 전달할 인자들을 동적으로 구성.
    """
    model_args = {
        'nb_classes': config['nb_classes'],
        'Chans': chans,
        'Samples': samples,
        'kernLength': config.get('kernLength', 64),
        'class_weight': config.get('class_weight'),
        'F1': config.get('F1', 8),
        'D': config.get('D', 2),
        'F2': config.get('F2', 16),
    }

    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path is not None:
        model_args['checkpoint_path'] = checkpoint_path

    # 스케줄러 관련 설정
    scheduler_name = config.get('scheduler')
    if scheduler_name:
        model_args['scheduler_name'] = scheduler_name
        scheduler_keys = ['T_0', 'T_mult', 'T_up', 'gamma', 'eta_max', 'factor', 'patience', 'threshold','cooldown']
        for key in scheduler_keys:
            param = config.get(key)
            if param is not None:
                model_args[key] = param

    # 기타 설정
    model_args['lr'] = config.get('lr', 1e-3)
    model_args['dropoutRate'] = config.get('dropoutRate', 0.5)

    grl_lambda = config.get('grl_lambda', None)
    if grl_lambda is not None:
        model_args['grl_lambda'] = grl_lambda
        model_args['lnl_lambda'] = config.get('lnl_lambda', 0.01)

    domain_classes = config.get('domain_classes', None)
    if domain_classes is not None:
        if 'group' in config['data_config']:
            if 'UNMconfig' in config['data_config']:
                # 34, 35, 35
                if 'A' == config['subject_name']:
                    domain_classes = 34
                else:
                    domain_classes = 35
            elif 'UIconfig' in config['data_config']:
                #18, 18, 20
                if 'C' == config['subject_name']:
                    domain_classes = 20
                else:
                    domain_classes = 18
        model_args['domain_classes'] = domain_classes
    
    # groupDRO 관련 설정
    if n_groups is not None:
        model_args['n_groups'] = n_groups
        if group_counts is not None:
            model_args['group_counts'] = group_counts
        else:
            raise ValueError("group_counts must be provided if n_groups is specified.")
    adj_constant = config['adj_constant']
    if adj_constant is not None:
        model_args['adj'] = adj_constant
    else:
        raise ValueError("adj_constant must be provided in the config.")

    return model_args

class LightningEEGDARunnerDRO(LightningEEGDARunner):
    """
    하나의 클래스에서 학습(train), 테스트(test) 로직을 모두 수행.
    """
    def __init__(self, config: dict, checkpoint: str = None):
        """
        :param config: 학습/테스트에 필요한 설정 딕셔너리
        :param checkpoint: 테스트 시 사용할 체크포인트 경로(디렉토리) (학습 시에는 None 가능)
        """
        super().__init__(config, checkpoint)

    # --------------------------
    # Train Internals
    # --------------------------
    def _build_data_module_for_train(self):
        """
        훈련/검증 세트 구성을 위한 DataModule 생성 및 setup('fit')
        """
        config = self.config
        skip_time_list = None
        if config.get("skip_time_list", None):
            with open(config["skip_time_list"] + ".json", 'r') as f:
                skip_time_list = json.load(f)

        # data_config json 로드
        data_config_path = config["data_config"] + config.get('subject_name', "") + ".json"
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)

        print(f"Data config loaded from: {data_config_path}")
        self.data_module = EEGDataModuleGroupDRO(
            data_config=data_config,
            batch_size=config['batch_size'],
            masking_ch_list=config['masking_ch_list'],
            rm_ch_list=config['rm_ch_list'],
            subject_usage=config['subject_usage'],
            seed=config['seed'] if config.get('fix_seed', False) else None,
            default_path=config['data_default_path'],
            skip_time_list=skip_time_list
        )
        
        print("starting setup for training data module...")
        self.data_module.setup('fit')

    def _build_model_for_train(self):
        """
        학습용 모델 인스턴스 생성
        """
        model_dict = {
            'EEGNetDRO': EEGNetDRO
        }
        model_class = model_dict[self.config['model_name']]

        chans = self.data_module.chans
        samples = self.data_module.samples
        n_groups = self.data_module.get_num_groups()
        group_counts = self.data_module.get_group_counts()
        print(f"Channels (Chans): {chans}")
        print(f"Samples (Samples): {samples}")
        print(f"Number of groups (n_groups): {n_groups}")
        print(f"Group counts (group_counts): {group_counts}")

        model_args = build_model_args(self.config, chans, samples, n_groups, group_counts)
        print(">> model_args for train:", model_args)
        self.model = model_class(**model_args)

    # --------------------------
    # Test Internals
    # --------------------------
    def _build_data_module_for_test(self, sub_config_path, test_data_default_path):
        """
        테스트 세트 구성을 위한 DataModule 생성 및 setup('test')
        (원본 test_func의 path 보정 로직 포함)
        """
        config = self.config
        # 구버전 base path 보정 로직 (원본 test_func 참고)
        # config_path = config["data_config"] + config.get('subject_name', "") + ".json"
        config_path = sub_config_path
        old_base = "/mnt/sdb1/jsw/hanseul/code/Fairness_for_generalization"
        new_base = "/home/jsw/Fairness/Fairness_for_generalization"
        if config_path.startswith(old_base):
            relative_path = os.path.relpath(config_path, old_base)
            config_path = os.path.join(new_base, relative_path)
            print(f"Corrected test config_path: {config_path}")
        else:
            # 필요시 로깅
            pass

        # skip_time_list 로드
        skip_time_list = None
        if config.get("skip_time_list", None):
            with open(config["skip_time_list"] + ".json", 'r') as f:
                skip_time_list = json.load(f)

        with open(config_path, 'r') as f:
            data_config = json.load(f)

        self.data_module = EEGDataModuleGroupDRO(
            data_config=data_config,
            batch_size=config['batch_size'],
            masking_ch_list=config['masking_ch_list'],
            rm_ch_list=config['rm_ch_list'],
            subject_usage=config['subject_usage'],
            seed=config['seed'] if config.get('fix_seed', False) else None,
            default_path=test_data_default_path,
            skip_time_list=skip_time_list
        )
        self.data_module.setup('test')

    def _build_model_for_test(self):
        """
        체크포인트로부터 모델 로드
        """
        if not self.checkpoint:
            raise ValueError("테스트를 위해서는 checkpoint 경로가 필요합니다.")

        model_dict = {
            'EEGNetDRO': EEGNetDRO
        }
        model_class = model_dict[self.config['model_name']]

        ckpt_file = os.path.join(self.checkpoint, "checkpoint.ckpt")
        print(f"Loading model from checkpoint: {ckpt_file}")
        self.model = model_class.load_from_checkpoint(ckpt_file)


# ----------------------------------------------------------------------------
# 최종적으로 train_func, test_func를 제공
# ----------------------------------------------------------------------------
def train_func(config):
    """
    Ray Tune 등에서 사용하기 위한 학습 함수.
    하지만 groupDRO를 첨가한
    """
    print("Starting training with groupDRO...")
    runner = LightningEEGDARunnerDRO(config, checkpoint=None)
    runner.train()
    # 학습 시 RayTrainReportCallback이 내부적으로 성능 지표를 반환하므로
    # 여기서는 별도 return이 없어도 됩니다.


def test_func(config, checkpoint, sub_config_path, test_data_default_path):
    """
    Ray Tune 등에서 사용하기 위한 테스트 함수.
    하지만 groupDRO를 첨가한
    """
    print("\U0001F6A8"*50)
    print("Starting testing with groupDRO...")
    print("\U0001F6A8"*50)
    runner = LightningEEGDARunnerDRO(config, checkpoint=checkpoint)
    test_results = runner.test(sub_config_path, test_data_default_path)
    return test_results
