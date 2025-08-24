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
from src.models import EEGNet
from src.models.eegnet_grl import EEGNetGRL, EEGNetLNL, EEGNetMI, EEGNetLNLAutoCorrelation
from src.models.eegnetDRO import EEGNetDRO
from src.models.eegnet_grl_lag import EEGNetLNLLag
from src.data.modules.EEGdataModuel import EEGDataModule
import yaml


def build_model_args(config, chans, samples):
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

    return model_args


class LightningEEGDARunner:
    """
    하나의 클래스에서 학습(train), 테스트(test) 로직을 모두 수행.
    """
    def __init__(self, config: dict, checkpoint: str = None):
        """
        :param config: 학습/테스트에 필요한 설정 딕셔너리
        :param checkpoint: 테스트 시 사용할 체크포인트 경로(디렉토리) (학습 시에는 None 가능)
        """
        self.config = config
        self.checkpoint = checkpoint  # 테스트 시 필요한 ckpt 경로
        self.data_module = None
        self.model = None
        self.trainer = None

        # 시드 세팅
        if config.get("fix_seed", False):
            self._set_seed(config["seed"])

    def train(self):
        """
        학습 파이프라인:
         1) 데이터 모듈 생성 및 setup('fit')
         2) 모델 생성
         3) Trainer 생성 (Ray DDP 등 적용)
         4) trainer.fit
        """
        self._build_data_module_for_train()
        self._build_model_for_train()
        self._create_trainer_for_train()
        self._run_train()

    def test(self, sub_config_path, test_data_default_path):
        """
        테스트 파이프라인:
         1) 데이터 모듈 생성 및 setup('test')
         2) 체크포인트를 이용해 모델 생성
         3) 일반 Trainer 생성
         4) trainer.test
        :return: 테스트 결과
        """
        self._build_data_module_for_test(sub_config_path, test_data_default_path)
        self._build_model_for_test()
        self._create_trainer_for_test()
        return self._run_test()

    def _set_seed(self, seed):
        """
        시드를 설정하는 함수
        """
        print(f"Setting all random seeds to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_everything(seed, workers=True)

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

        self.data_module = EEGDataModule(
            data_config=data_config,
            batch_size=config['batch_size'],
            masking_ch_list=config['masking_ch_list'],
            rm_ch_list=config['rm_ch_list'],
            subject_usage=config['subject_usage'],
            seed=config['seed'] if config.get('fix_seed', False) else None,
            default_path=config['data_default_path'],
            skip_time_list=skip_time_list,
            data_augmentation_config=self._create_data_augmentation_config()
        )

        print("starting setup for training data module...")
        self.data_module.setup('fit')

    def _build_model_for_train(self):
        """
        학습용 모델 인스턴스 생성
        """
        model_dict = {
            'EEGNet': EEGNet,
            'EEGNetDomainAdaptation_LNL': EEGNetLNL,
            'EEGNetDomainAdaptation_Not_GRL': EEGNetMI,
            'EEGNetDomainAdaptation_Only_GRL': EEGNetGRL,
            'EEGNetLNL': EEGNetLNL,
            'EEGNetMI': EEGNetMI,
            'EEGNetGRL': EEGNetGRL,
            'EEGNetLNLLag': EEGNetLNLLag,
            'EEGNetLNLAutoCorrelation': EEGNetLNLAutoCorrelation
        }
        model_class = model_dict[self.config['model_name']]

        chans = self.data_module.chans
        samples = self.data_module.samples
        print(f"Channels (Chans): {chans}")
        print(f"Samples (Samples): {samples}")

        model_args = build_model_args(self.config, chans, samples)
        print(">> model_args for train:", model_args)
        self.model = model_class(**model_args)
    
    def _load_cortical_regions(self, regions_path):
        """
        cortial_regions.json 파일을 로드하여 반환
        :param regions_path: regions.json 파일 경로
        :return: LIST[LIST[int]] 형태의 cortical regions
        """
        if regions_path is None:
            raise ValueError("Cortical regions path is required for data augmentation.")
        
        with open(regions_path, 'r') as f:
            regions = json.load(f)
        
        regions_list = []
        for key, values in regions.items():
            regions_list.append(values)

        return regions_list

    def _prepare_augmentation_config_for_datamodule(self, da_config):
        """
        EEGDataModule에 전달할 데이터 증강 설정을 최종적으로 구성합니다.
        YAML에서 로드된 설정과, 필요한 경우 추가적으로 처리된 데이터를 조합합니다.

        :param da_config: YAML 파일에서 로드된 데이터 증강 설정
        :return: EEGDataModule의 data_augmentation_config 파라미터에 전달될 최종 설정 딕셔너리
        """
        # map_search_space를 통해 처리된 설정 값들을 가져옵니다.
        # 예를 들어, da_config는 다음과 같은 형태일 수 있습니다:
        # {'train_only': True, 'methods': ['CorticalRegionChannelSwap'], 
        #  'setting': {'CorticalRegionChannelSwap': {...}, 'SubjectLevelChannelSwap': {...}}}
        
        final_config = {
            'enabled': da_config.get('enabled', True),
            'train_only': da_config.get('train_only', True),
            'methods': []
        }

        # 설정된 메서드 목록을 순회하며 필요한 정보를 구성합니다.
        for method_name in da_config.get('methods', []):
            method_setting = da_config.get('setting', {}).get(method_name)
            if not method_setting:
                print(f"Warning: Augmentation method '{method_name}' has no setting. Skipping.")
                continue

            method_info = {
                'type': method_setting.get('name'), # 'cortical' 또는 'subject'
                'prob_method': method_setting.get('swap_probability_method', 'uniform')
            }

            # CorticalRegionChannelSwap의 경우, regions 정보가 필요합니다.
            if method_name == 'CorticalRegionChannelSwap':
                regions_path = method_setting.get('cortical_regions_path')
                if not regions_path:
                    raise ValueError("cortical_regions_path is required for CorticalRegionChannelSwap.")
                
                # _load_cortical_regions를 통해 로드된 regions 정보를 사용합니다.
                # 실제 경로는 default_path와 결합하여 사용합니다.
                full_regions_path = os.path.join(self.config['default_path'], regions_path)
                method_info['regions'] = self._load_cortical_regions(full_regions_path)

            final_config['methods'].append(method_info)
            
        return final_config

    def _create_data_augmentation_config(self):
        """
        데이터 증강 설정을 위한 config 생성
        1. 메인 config에서 data_augmentation 설정을 가져옵니다.
        2. 활성화된 경우, config_path에 지정된 YAML 파일을 로드합니다.
        3. 로드된 설정을 EEGDataModule에 적합한 형태로 변환합니다.
        """
        da_config = self.config.get('data_augmentation', {})
        print(">> Initial data_augmentation config:", da_config)

        if not da_config.get('enabled', False):
            return {'enabled': False} # 데이터 증강 비활성화

        # data_augmentation.config_path에 지정된 YAML 파일 로드
        da_config_path = da_config.get('config_path')
        if not da_config_path:
            raise ValueError("Data augmentation is enabled, but 'config_path' is not specified.")
        
        full_config_path = os.path.join(self.config['default_path'], da_config_path)
        print(f">> Loading data augmentation config from: {full_config_path}")
        with open(full_config_path, 'r') as f:
            loaded_da_config = yaml.safe_load(f)

        # map_search_space를 사용하여 Ray Tune의 탐색 공간을 실제 값으로 변환
        from src.utils.run_utils.load_config import map_search_space
        mapped_da_config = map_search_space(loaded_da_config)
        
        # EEGDataModule에 전달하기 위해 최종적으로 설정 포맷팅
        final_da_config = self._prepare_augmentation_config_for_datamodule(mapped_da_config)
        final_da_config['enabled'] = True # 최종적으로 증강 활성화 상태 명시

        print(">> Final data_augmentation_config for DataModule:", final_da_config)
        return final_da_config

    def _create_trainer_for_train(self):
        """
        학습용 Trainer 생성 (Ray DDP, RayLightningEnvironment 적용)
        """
        lr_logger = LearningRateMonitor(logging_interval='epoch')
        callbacks = [RayTrainReportCallback(), lr_logger]
        early_stop_callback = EarlyStopping(
            monitor=self.config['monitor_value_name'],
            patience=10,
            mode='max',     # 기본값 'min'
            verbose=True, # 기본값 True
            min_delta=1e-3, # 개선으로 간주할 최소 변화량
            check_finite=True # loss가 NaN/inf인지 확인
            # 필요시 다른 EarlyStopping 파라미터 추가 가능
        )
        callbacks.append(early_stop_callback)


        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=callbacks,
            plugins=[RayLightningEnvironment()],
            logger=None,
            enable_progress_bar=False,
            min_epochs=20
        )
        # Ray Trainer 래핑
        self.trainer = prepare_trainer(trainer)

    def _run_train(self):
        """Trainer.fit 실행"""
        print(f"Train set size: {len(self.data_module.train_dataset)}")
        if hasattr(self.data_module, 'val_dataset'):
            print(f"Validation set size: {len(self.data_module.val_dataset)}")
        print("\n" + "⚡"*50 + "\n")
        print("Starting training phase...")
        self.trainer.fit(self.model, datamodule=self.data_module)
        print("Training phase completed.")

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

        self.data_module = EEGDataModule(
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
            'EEGNet': EEGNet,
            'EEGNetDomainAdaptation_LNL': EEGNetLNL,
            'EEGNetDomainAdaptation_Not_GRL': EEGNetMI,
            'EEGNetDomainAdaptation_Only_GRL': EEGNetGRL,
            'EEGNetLNL': EEGNetLNL,
            'EEGNetMI': EEGNetMI,
            'EEGNetGRL': EEGNetGRL,
            'EEGNetLNLLag': EEGNetLNLLag,
            "EEGNetDRO": EEGNetDRO
        }
        model_class = model_dict[self.config['model_name']]

        ckpt_file = os.path.join(self.checkpoint, "checkpoint.ckpt")
        print(f"Loading model from checkpoint: {ckpt_file}")
        self.model = model_class.load_from_checkpoint(ckpt_file)

    def _create_trainer_for_test(self):
        """
        테스트용 Trainer 생성 (일반 Lightning Trainer)
        """
        self.trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            logger=None,
            enable_progress_bar=False
        )

    def _run_test(self):
        """테스트 실행 및 결과 반환"""
        if hasattr(self.data_module, 'test_dataset'):
            print(f"Test set size: {len(self.data_module.test_dataset)}")
        print("\n" + "⚡"*50 + "\n")
        print("Starting testing phase...")
        test_results = self.trainer.test(self.model, datamodule=self.data_module)
        print("Testing phase completed.")
        print(f"Test Results: {test_results}")
        return test_results


# ----------------------------------------------------------------------------
# 최종적으로 train_func, test_func를 제공
# ----------------------------------------------------------------------------
def train_func(config):
    """
    Ray Tune 등에서 사용하기 위한 학습 함수.
    """
    runner = LightningEEGDARunner(config, checkpoint=None)
    runner.train()
    # 학습 시 RayTrainReportCallback이 내부적으로 성능 지표를 반환하므로
    # 여기서는 별도 return이 없어도 됩니다.


def test_func(config, checkpoint, sub_config_path, test_data_default_path):
    """
    Ray Tune 등에서 사용하기 위한 테스트 함수.
    """
    runner = LightningEEGDARunner(config, checkpoint=checkpoint)
    test_results = runner.test(sub_config_path, test_data_default_path)
    return test_results
