from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class EEGDataset(Dataset):
    def __init__(self, data_x, data_y, domain_y=None, kernels=1):
        self.data_x = data_x
        self.data_y = data_y
        self.domain_y = domain_y
        self.kernels = kernels

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        x = x[:, :, np.newaxis]
        x = x.reshape(x.shape[0], x.shape[1], self.kernels)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        if self.domain_y is not None:
            domain_label = torch.tensor(self.domain_y[idx], dtype=torch.long)
            return x, (y, domain_label)
        else:
            return x, y

"""
train_subject : all, test1

"""
class DataProvider:
    def __init__(self, data_config, masking_ch_list=None, rm_ch_list=None, subject_usage="all"):
        self.data_config = data_config
        self.masking_ch_list = masking_ch_list if masking_ch_list else []
        self.rm_ch_list = rm_ch_list if rm_ch_list else []

        # 로드 및 데이터 준비
        load_and_prepare_data_func = self.call_load_and_prepare_data(subject_usage)
        self.data_x, self.data_y, self.domain_y = load_and_prepare_data_func()
        self.chans = self.data_x.shape[1]
        self.samples = self.data_x.shape[2]

        # 클래스 수와 레이블 매핑 설정
        self.unique_labels = sorted(set(self.data_y))
        self.num_classes = len(self.unique_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.data_y = np.vectorize(self.label_to_index.get)(self.data_y)

        if self.domain_y is not None:
            self.domain_unique_labels = sorted(set(self.domain_y))
            self.num_domain_classes = len(self.domain_unique_labels)
            self.domain_label_to_index = {label: idx for idx, label in enumerate(self.domain_unique_labels)}
            self.domain_y = np.vectorize(self.domain_label_to_index.get)(self.domain_y)

    def call_load_and_prepare_data(self, subject_usage):
        return getattr(self, "load_and_prepare_data_"+subject_usage)

    def load_and_prepare_data_all(self):
        data_x = []
        data_y = []
        
        domain_y = [] if 'domain_list' in self.data_config else None

        # 데이터 리스트와 도메인 리스트에서 파일 경로 매핑 생성
        file_to_label = {}
        for split in self.data_config['data_list']:
            for label in self.data_config['data_list'][split]:
                for file_path in self.data_config['data_list'][split][label]:
                    file_to_label[file_path] = label

        file_to_domain = {}
        if 'domain_list' in self.data_config:
            for split in self.data_config['domain_list']:
                for domain_label in self.data_config['domain_list'][split]:
                    for file_path in self.data_config['domain_list'][split][domain_label]:
                        file_to_domain[file_path] = domain_label

        # 데이터 로딩 및 레이블 할당
        loaded_files = set()
        for file_path, label in file_to_label.items():
            if file_path in loaded_files:
                continue  # 이미 로드된 파일은 건너뜁니다.
            data = np.load(file_path)

            # # 스킵 리스트 적용 (옵션)
            # skip_indices = self.get_skip_indices(file_path)
            # if skip_indices:
            #     data = np.delete(data, skip_indices, axis=0)

            # 채널 마스킹 및 제거
            data = self.apply_channel_modifications(data)

            # 데이터 및 레이블 추가
            for sample in data:
                data_x.append(sample)
                data_y.append(label)
                if 'domain_list' in self.data_config:
                    domain_label = file_to_domain.get(file_path, None)
                    domain_y.append(domain_label)

            loaded_files.add(file_path)

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        if 'domain_list' in self.data_config:
            domain_y = np.array(domain_y) if domain_y else None

        return data_x, data_y, domain_y
    
    def load_and_prepare_data_test1(self):
        data_x = []
        data_y = []
        
        domain_y = [] if 'domain_list' in self.data_config else None

        # 데이터 리스트와 도메인 리스트에서 파일 경로 매핑 생성
        file_to_label = {}
        file_to_subject_usage = {}
        for split in self.data_config['data_list']:
            for label in self.data_config['data_list'][split]:
                for file_path in self.data_config['data_list'][split][label]:
                    file_to_subject_usage[file_path] = split
                    file_to_label[file_path] = label

        file_to_domain = {}
        if 'domain_list' in self.data_config:
            for split in self.data_config['domain_list']:
                for domain_label in self.data_config['domain_list'][split]:
                    for file_path in self.data_config['domain_list'][split][domain_label]:
                        file_to_subject_usage[file_path] = split
                        file_to_domain[file_path] = domain_label

        # 데이터 로딩 및 레이블 할당
        loaded_files = set()
        for file_path, label in file_to_label.items():
            if file_path in loaded_files:
                continue  # 이미 로드된 파일은 건너뜁니다.
            data = np.load(file_path)

            # # 스킵 리스트 적용 (옵션)
            # skip_indices = self.get_skip_indices(file_path)
            # if skip_indices:
            #     data = np.delete(data, skip_indices, axis=0)

            # 채널 마스킹 및 제거
            data = self.apply_channel_modifications(data)

            # 데이터 및 레이블 추가
            for sample in data:
                data_x.append(sample)
                data_y.append(label)
                if 'domain_list' in self.data_config:
                    domain_label = file_to_domain.get(file_path, None)
                    domain_y.append(domain_label)

            loaded_files.add(file_path)

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        if 'domain_list' in self.data_config:
            domain_y = np.array(domain_y) if domain_y else None

        return data_x, data_y, domain_y

    def get_skip_indices(self, file_path):
        # 스킵 리스트에서 해당 파일의 스킵 인덱스를 가져옵니다.
        skip_indices = []
        for split in self.data_config.get('skip_list', {}):
            for label in self.data_config['skip_list'][split]:
                for idx, path in enumerate(self.data_config['data_list'][split][label]):
                    if path == file_path:
                        skip_indices = self.data_config['skip_list'][split][label].get(str(idx), [])
                        return skip_indices
        return skip_indices

    def apply_channel_modifications(self, data):
        # 채널 마스킹
        if self.masking_ch_list:
            for n in range(len(data)):
                for mask_ch in self.masking_ch_list:
                    if mask_ch < data.shape[1]:
                        data[n][mask_ch] = np.zeros_like(data[n][mask_ch])

        # 채널 제거
        if self.rm_ch_list:
            rm_ch_valid = [ch for ch in self.rm_ch_list if ch < data.shape[1]]
            data = np.delete(data, rm_ch_valid, axis=1)

        return data

    def get_dataset(self):
        if self.domain_y is not None:
            return EEGDataset(self.data_x, self.data_y, self.domain_y)
        else:
            return EEGDataset(self.data_x, self.data_y)

    def get_data_shape(self):
        return self.chans, self.samples

    def print_sample(self, idx=0):
        """특정 인덱스의 샘플을 출력합니다 (기본 인덱스는 0)."""
        dataset = self.get_dataset()
        sample = dataset[idx]
        if self.domain_y is not None:
            sample_x, (sample_y, sample_domain_y) = sample
            print(f"Sample {idx} - X shape: {sample_x.shape}, Y: {sample_y}, Domain Y: {sample_domain_y}")
        else:
            sample_x, sample_y = sample
            print(f"Sample {idx} - X shape: {sample_x.shape}, Y: {sample_y}")