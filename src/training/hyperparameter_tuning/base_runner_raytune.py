from abc import ABC, abstractmethod

from src.utils.run_utils.load_config import load_ray_tune_config, map_search_space, setup_path

class BaseTuneRunner(ABC):
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.tune_result = None

    def tunner_setup(self):
        self.load_config()
        self.setup_path()
        self.create_search_space()
        self.create_scheduler()
        self.create_reporter()
        self.create_scaling_config()
        self.create_run_config()
        self.create_tune_config()
        self.create_trainer()
    
    def restore(self):
        """ 전체 실행 파이프라인 """
        self.load_config()
        self.setup_path()
        self.create_search_space()
        self.create_scheduler()
        self.create_reporter()
        self.create_scaling_config()
        self.create_run_config()
        self.create_tune_config()
        self.create_trainer()        
        self.restore_tuner()


    def run(self):
        """ 전체 실행 파이프라인 """
        self.tunner_setup()
        self.run_tuner()

    def load_config(self):
        """Ray Tune 설정 파일(YAML 등)을 로드"""
        self.config = load_ray_tune_config(self.config_path)
        self.ray_tune_config = self.config['ray_tune']

    def setup_path(self):
        """사용할 경로 등 사전 준비"""
        setup_path(self.ray_tune_config)

    def create_search_space(self):
        """Search Space 생성"""
        self.search_space = map_search_space(self.ray_tune_config['search_space'])
        print("Search Space:", self.search_space)

    @abstractmethod
    def create_scheduler(self):
        """스케줄러 생성"""
        pass

    @abstractmethod
    def create_reporter(self):
        """Reporter 생성"""
        pass

    @abstractmethod
    def create_scaling_config(self):
        """ScalingConfig 생성"""
        pass

    @abstractmethod
    def create_run_config(self):
        """RunConfig 생성"""
        pass

    @abstractmethod
    def create_tune_config(self):
        """TuneConfig 생성"""
        pass

    @abstractmethod
    def create_trainer(self):
        """Trainer(TorchTrainer 등) 생성"""
        pass

    @abstractmethod
    def run_tuner(self):
        """Tuner 생성 및 실행"""
        pass

    def restore_tuner(self):
        """Tuner 생성 및 재개"""
        pass