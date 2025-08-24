import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import math

class EEGDatasetGroupDRO(Dataset):
    def __init__(self, data_x, data_y, domain_y=None, kernels=1, group_y=None): # Add group_y
        self.data_x = data_x
        self.data_y = data_y
        self.domain_y = domain_y
        self.group_y = group_y # Store group_y
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
        domain_label = torch.tensor(self.domain_y[idx], dtype=torch.long)
        group_label = torch.tensor(self.group_y[idx], dtype=torch.long)

        # Return y, domain_label, and group_label
        return x, (y, domain_label, group_label)


class EEGDataModuleGroupDRO(pl.LightningDataModule):
    def __init__(self, data_config: dict, batch_size: int = 16, masking_ch_list=None, rm_ch_list=None, subject_usage: str = "all", seed: int = 42, skip_time_list:dict=None, default_path:str="/home/jsw/Fairness/Fairness_for_generalization"):
        super().__init__()
        self.data_config = data_config
        self.skip_time_list=skip_time_list
        self.batch_size = batch_size
        self.masking_ch_list = masking_ch_list if masking_ch_list else []
        self.rm_ch_list = rm_ch_list if rm_ch_list else []
        self.subject_usage = subject_usage
        self.seed = seed        
        self.default_path=default_path
        
        # 데이터를 초기화 단계에서 준비하여 setup 호출 시 반복 작업 방지
        self.data_x, self.data_y, self.domain_y = self.load_and_prepare_dataDict()
        self.get_info_from_data()
        self.create_confounder_info()
        # self.get_unique_label_counts()

        # 데이터셋 분할 초기화에서 수행
        if not any(self.data_config['data_list']['test'].values()):
            # test 데이터가 없을 경우 train 데이터를 나눠서 train, val, test로 구성
            train_files, temp_files, train_labels, temp_labels, \
            train_domain_labels, temp_domain_labels, train_group_labels, temp_group_labels = train_test_split(
                self.data_x['train'],
                self.data_y['train'],
                self.domain_y['train'] if self.domain_y is not None else [None] * len(self.data_y['train']),
                self.group_y['train'] if hasattr(self, 'group_y') and self.group_y is not None else [None] * len(self.data_y['train']),
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
                stratify=self.data_y['train']
            )
            val_files, test_files, val_labels, test_labels, \
            val_domain_labels, test_domain_labels, val_group_labels, test_group_labels = train_test_split(
                temp_files,
                temp_labels,
                temp_domain_labels,
                temp_group_labels,
                test_size=0.5,
                random_state=self.seed,
                shuffle=True,
                stratify=temp_labels
            )

            self.data_x['train'] = train_files
            self.data_x['val'] = val_files
            self.data_x['test'] = test_files

            self.data_y['train'] = train_labels
            self.data_y['val'] = val_labels
            self.data_y['test'] = test_labels

            if self.domain_y is not None:
                self.domain_y['train'] = train_domain_labels
                self.domain_y['val'] = val_domain_labels
                self.domain_y['test'] = test_domain_labels
            if self.group_y is not None:
                self.group_y['train'] = train_group_labels
                self.group_y['val'] = val_group_labels
                self.group_y['test'] = test_group_labels
        else:
            # test 데이터가 있을 경우 train 데이터를 train, val로 나눔
            train_files, val_files, train_labels, val_labels,\
            train_domain_labels, val_domain_labels, train_group_labels, val_group_labels = train_test_split(
                self.data_x['train'],
                self.data_y['train'],
                self.domain_y['train'] if self.domain_y is not None else [None] * len(self.data_y['train']),
                self.group_y['train'] if hasattr(self, 'group_y') and self.group_y is not None else [None] * len(self.data_y['train']),
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
                stratify=self.data_y['train']
            )

            self.data_x['train'] = train_files
            self.data_x['val'] = val_files

            self.data_y['train'] = train_labels
            self.data_y['val'] = val_labels

            if self.domain_y is not None:
                self.domain_y['train'] = train_domain_labels
                self.domain_y['val'] = val_domain_labels
            if self.group_y is not None:
                self.group_y['train'] = train_group_labels
                self.group_y['val'] = val_group_labels
        self.get_unique_label_counts()

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.setup_fit()
        if stage == 'validate' or stage is None:
            self.setup_validate()
        if stage == 'test' or stage is None:
            self.setup_test()
        if stage == 'predict' or stage is None:
            self.setup_predict()

    def setup_fit(self):
        self.train_dataset = EEGDatasetGroupDRO(
            data_x=self.data_x['train'],
            data_y=self.data_y['train'],
            domain_y=self.domain_y['train'] if self.domain_y is not None else None,
            group_y=self.group_y['train'] if hasattr(self, 'group_y') and self.group_y is not None else None, # Pass group_y
            kernels=1
        )
        self.val_dataset = EEGDatasetGroupDRO(
            data_x=self.data_x['val'],
            data_y=self.data_y['val'],
            domain_y=self.domain_y['val'] if self.domain_y is not None else None,
            group_y=self.group_y['val'] if hasattr(self, 'group_y') and self.group_y is not None else None, # Pass group_y
            kernels=1
        )

    def setup_validate(self):
        self.val_dataset = EEGDatasetGroupDRO(
            data_x=self.data_x['val'],
            data_y=self.data_y['val'],
            domain_y=self.domain_y['val'] if self.domain_y is not None else None,
            group_y=self.group_y['val'] if hasattr(self, 'group_y') and self.group_y is not None else None, # Pass group_y
            kernels=1
        )

    def setup_test(self):
        self.test_dataset = EEGDatasetGroupDRO(
            data_x=self.data_x['test'],
            data_y=self.data_y['test'],
            domain_y=self.domain_y['test'] if self.domain_y is not None and self.subject_usage == 'all' else None,
            group_y=self.group_y['test'] if hasattr(self, 'group_y') and self.group_y is not None else None, # Pass group_y
            kernels=1
        )

    def setup_predict(self):
        self.predict_dataset = EEGDatasetGroupDRO(
            data_x=self.data_x['test'],  # 예측 시 테스트 데이터 사용
            data_y=self.data_y['test'],
            domain_y=self.domain_y['test'] if self.domain_y is not None and self.subject_usage == 'all' else None,
            group_y=self.group_y['test'] if hasattr(self, 'group_y') and self.group_y is not None else None, # Pass group_y
            kernels=1
        )

    def train_dataloader(self):
        # Training and reweighting
        # When the --robust flag is not set, reweighting changes the loss function
        # from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
        # When the --robust flag is set, reweighting does not change the loss function
        # since the minibatch is only used for mean gradient estimation for each group separately
        
        # train split의 데이터 총 개수
        data_len = len(self.data_x['train'])
        for split in ['train']:
            group_weights = data_len / self._group_counts[split]
            weights = group_weights[self.group_y[split]]

        

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, data_len, replacement=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
    
    def load_and_prepare_dataDict(self):
        data_x = {'train': [], 'val': [], 'test': []}
        data_y = {'train': [], 'val': [], 'test': []}
        domain_y = {'train': [], 'val': [], 'test': []} if 'domain_list' in self.data_config else None

        def match_path2label(save_diction: dict, list_name: str) -> None:
            for split in self.data_config[list_name]:
                for label in self.data_config[list_name][split]:
                    for file_path in self.data_config[list_name][split][label]:
                        # file_path는 객체를 구분하는 단위로써 사용
                        save_diction[split][os.path.join(self.default_path,file_path)] = label

        def apply_channel_modifications(data) -> None:
            # 채널 마스킹
            if self.masking_ch_list:
                for n in range(len(data)):
                    for mask_ch in self.masking_ch_list:
                        if mask_ch < data.shape[1]:
                            data[n][mask_ch] = np.zeros_like(data[n][mask_ch])
            return data
        
        def apply_channel_remove(data) -> None:
            # 채널 마스킹
            if self.rm_ch_list:
                data = np.delete(data,self.rm_ch_list,axis=1)
            return data
        
        def apply_skip_time(data, beforeOrAfter:str, subjectName:str) -> np.ndarray:
            print('>> do skip?')
            if self.skip_time_list:
                print('>> Yes!!')
                print(beforeOrAfter)
                print(subjectName)
                for skip_range in self.skip_time_list.get(beforeOrAfter).get(subjectName):
                    print(f">> skip_range : {skip_range}")
                    if skip_range:
                        print(f"befo shape : {data.shape}")
                        data = np.delete(data, np.s_[math.floor(skip_range[0] / 6) + 1 :  math.ceil(skip_range[1] / 6) + 1], axis=0)
                        print(f"aft shape : {data.shape}")
                        print(">> deleted")
                return data
            else:
                print(">> Nope")
                return data

        file_to_label = {'train': {}, 'val': {}, 'test': {}}
        file_to_domain = {'train': {}, 'val': {}, 'test': {}} if 'domain_list' in self.data_config else None        
        match_path2label(file_to_label, "data_list")
        if 'domain_list' in self.data_config:
            match_path2label(file_to_domain, "domain_list")
        for split in ['train', 'val', 'test']:
            for file_path, label in file_to_label.get(split, {}).items():
                if file_path is None or label is None:
                    continue
                try:
                    print(">> data loading...")
                    data = np.load(file_path)
                    print(">> successful...")
                    print(f">> data shape : {data.shape}")
                    print(f">> label value : {label}")
                except FileNotFoundError:
                    print(f"파일을 찾을 수 없습니다: {file_path}")
                    continue
                except Exception as e:
                    print(f"파일을 로드하는 중 오류 발생 ({file_path}): {e}")
                    continue

                data = apply_channel_modifications(data)

                data = apply_channel_remove(data)

                attrList = file_path.split('_')[-3:]
                data = apply_skip_time(data, beforeOrAfter=attrList[1], subjectName=attrList[0].split('/')[-1])

                tmp_data_x = []
                tmp_data_y = []
                tmp_domain_y = []
                for sample in data:
                    tmp_data_x.append(sample)
                    tmp_data_y.append(label)
                    if 'domain_list' in self.data_config and file_to_domain[split]:
                        domain_label = file_to_domain[split].get(file_path, None)
                        tmp_domain_y.append(domain_label)

                data_x[split].extend(tmp_data_x)
                data_y[split].extend(tmp_data_y)
                if 'domain_list' in self.data_config and file_to_domain[split]:
                    domain_y[split].extend(tmp_domain_y)

            data_x[split] = np.array(data_x[split])
            data_y[split] = np.array(data_y[split])
            if 'domain_list' in self.data_config and file_to_domain[split]:
                domain_y[split] = np.array(domain_y[split])
        return data_x, data_y, domain_y

    def get_info_from_data(self):
        self.data_shape = self.data_x['train'].shape
        self.chans = self.data_shape[1]
        self.samples = self.data_shape[2]

        self.unique_labels = set()
        if self.domain_y is not None:
            self.domain_unique_labels = set()
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            self.unique_labels.update(np.unique(self.data_y[split]))
            if self.domain_y is not None and self.domain_y[split] is not None:
                self.domain_unique_labels.update(np.unique(self.domain_y[split]))

        self.unique_labels = sorted(self.unique_labels)
        self.num_classes = len(self.unique_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}

        if self.domain_y is not None:
            self.domain_unique_labels = sorted(self.domain_unique_labels)
            self.num_domain_classes = len(self.domain_unique_labels)
            self.domain_label_to_index = {label: idx for idx, label in enumerate(self.domain_unique_labels)}

        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            self.data_y[split] = np.array([self.label_to_index[label] for label in self.data_y[split]])
            if self.domain_y is not None:
                self.domain_y[split] = np.array([self.domain_label_to_index[label] for label in self.domain_y[split]])
    
    def create_confounder_info(self):
        if self.domain_y is None:
            raise ValueError("Domain labels are not available. Cannot create confounder info.")
        
        # self.attrs_df가 각 속성 값을 나타내는 열에 0,1 이진으로 인코딩되어 있고
        # 속성들이 겹쳐서 나타날 수 있는 경우를 가정합니다.
        # # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        # self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        # self.n_confounders = len(self.confounder_idx)
        # confounders = self.attrs_df[:, self.confounder_idx]
        # confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        # self.confounder_array = confounder_id

        # # Map to groups
        # self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        # self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')


        # 하지만 현재 domain인 subject는 confounder 중 오직 하나의 단일 속성만을 가지도록 설계
        group_y = {'train': [], 'val': [], 'test': []}
        # In this setting that the domain is a subject name, we assume that each subject is a single confounder.
        # And they only can have one confounder attribute.
        # So, we can simply map the domain label to a group label.
        # self.confounder_y = self.domain_y
        self.num_groups = self.num_classes * pow(2, self.num_domain_classes)
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            if split == 'test':
                continue
            print(f">> split : {split}, data_y shape : {self.data_y[split].shape}, domain_y shape : {self.domain_y[split].shape}")
            group_y[split] = (self.data_y[split] * (self.num_groups / 2) + self.domain_y[split]).astype('int')
        self.group_y = group_y
        self.group_unique_labels = sorted(set(np.concatenate([group_y[split] for split in ['train', 'val', 'test']])))
        self.total_num_group_classes = len(self.group_unique_labels)


    def get_unique_label_counts(self):
        # self.data_y의 각 split에 대해 고유한 레이블별 데이터 개수를 계산
        # self.group_y의 각 split에 대해 고유한 그룹 레이블별 데이터 개수를 계산
        # 해당 함수가 호출 되는 시점에서 'train', 'test' 이 key에만 데이터가 존재.
        # 'test'와 'train', 'val'은 서로 attibute를 공유하지 않으므로 'test'는 무시
        # 데이터 배분이 끝나고 호출해야 함.
        # 각 train, val의 counts를 새는 거기 때문임
        # n_groups === num_groups
        # n_classes === num_classes
        # self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        # self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

        self._y_counts = {split: {} for split in ['train', 'val', 'test']}
        self._group_counts = {split: {} for split in ['train', 'val', 'test']}
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            if split == 'test':
                continue
            self._group_counts[split] = (np.arange(self.num_groups)[:, None] == self.group_y[split]).sum(axis=1).astype(float)
            self._y_counts[split] = (np.arange(self.num_classes)[:, None] == self.data_y[split]).sum(axis=1).astype(float)

    def get_num_groups(self):
        if hasattr(self, 'group_y') and self.group_y is not None:
            return self.num_groups
        else:
            raise ValueError("Group labels are not available. Cannot get number of groups.")
    
    def get_group_counts(self):
        if hasattr(self, '_group_counts') and self._group_counts is not None:
            return self._group_counts
        else:
            raise ValueError("Group counts are not available. Cannot get group counts.")