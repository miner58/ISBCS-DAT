"""
DMMR Compatible DataModule

Zero-disruption extension of EEGDataModule to support DMMR training requirements.
Maintains 100% backward compatibility while adding DMMR-specific functionality.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import Counter
import pytorch_lightning as pl
import random
from collections import defaultdict

from .EEGdataModuel import EEGDataModule, EEGDataset

class SubjectAwareBalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, num_classes, num_subjects, shuffle=True, drop_last=True, sampler=None, use_dynamic_batch_size=False):
        # BatchSampler requires a sampler, batch_size, and drop_last
        from torch.utils.data import SequentialSampler
        if sampler is None:
            sampler = SequentialSampler(dataset)
        super().__init__(sampler, batch_size, drop_last=drop_last)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_subjects = num_subjects
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_dynamic_batch_size = use_dynamic_batch_size
        
        # Subject별 클래스별 인덱스 수집
        self.subject_class_indices = self._collect_subject_class_indices()
        
        # 배치 구성 전략 개선
        self._validate_data_availability()
        
        # 배치 크기 전략 선택
        if self.use_dynamic_batch_size:
            print("📐 Using dynamic batch size calculation (legacy mode)")
            self._calculate_dynamic_sampling_strategy()
        else:
            print("🎯 Using fixed batch size (forced to match config)")
            self._calculate_fixed_sampling_strategy()
        
        print(f"🎯 SubjectAwareBalancedBatchSampler initialized:")
        print(f"   Available combinations: {len(self.subject_class_indices)}")
        if self.use_dynamic_batch_size:
            print(f"   Expected per batch: {self.samples_per_subject_class} per (subject, class)")
            print(f"   Actual batch size: {len(self.subject_class_indices) * self.samples_per_subject_class}")
        else:
            print(f"   Fixed batch size: {self.batch_size}")
            print(f"   Allocation strategy: {self.allocation_strategy}")
        
    def _collect_subject_class_indices(self):
        """Subject별, 클래스별 인덱스 수집"""
        subject_class_indices = {}
        
        for idx in range(len(self.dataset)):
            _, _, label_tensor, _ = self.dataset[idx]
            
            # 각 subject별로 라벨 확인
            for subject_idx in range(self.num_subjects):
                if subject_idx < len(label_tensor):  # 안전성 체크
                    label = label_tensor[subject_idx].item()
                    
                    key = (subject_idx, label)
                    if key not in subject_class_indices:
                        subject_class_indices[key] = []
                    subject_class_indices[key].append(idx)
        
        return subject_class_indices
    
    def _validate_data_availability(self):
        """데이터 가용성 검증"""
        missing_combinations = []
        for subject_idx in range(self.num_subjects):
            for class_id in range(self.num_classes):
                key = (subject_idx, class_id)
                if key not in self.subject_class_indices:
                    missing_combinations.append(key)
        
        if missing_combinations:
            print(f"⚠️  Missing (subject, class) combinations: {missing_combinations}")
            print("   This may cause unbalanced batches")
    
    def _calculate_dynamic_sampling_strategy(self):
        """동적 배치 크기 계산 (기존 방식)"""
        available_combinations = len(self.subject_class_indices)
        
        if available_combinations == 0:
            raise ValueError("No subject-class combinations available")
        
        # 기존 방식: 조합 수에 따라 배치 크기가 동적으로 결정
        ideal_per_combination = self.batch_size // available_combinations
        self.samples_per_subject_class = max(1, ideal_per_combination)  # 최소 1개
        
        actual_batch_size = available_combinations * self.samples_per_subject_class
        print(f"📝 Dynamic batch calculation:")
        print(f"   Requested batch size: {self.batch_size}")
        print(f"   Actual batch size: {actual_batch_size}")
        print(f"   Samples per (subject, class): {self.samples_per_subject_class}")
    
    def _calculate_fixed_sampling_strategy(self):
        """고정 배치 크기 계산 (새로운 방식 - 기본값)"""
        available_combinations = len(self.subject_class_indices)
        
        if available_combinations == 0:
            raise ValueError("No subject-class combinations available")
        
        # 설정된 batch_size를 정확히 맞추는 할당 전략 계산
        base_allocation = self.batch_size // available_combinations
        extra_slots = self.batch_size % available_combinations
        
        # 각 조합별 할당량 결정
        self.allocation_strategy = []
        combination_keys = list(self.subject_class_indices.keys())
        
        for i in range(available_combinations):
            allocation = base_allocation
            # 나머지를 앞쪽 조합들에 순차적으로 분배
            if i < extra_slots:
                allocation += 1
            self.allocation_strategy.append((combination_keys[i], allocation))
        
        # 검증: 총 할당량이 batch_size와 일치하는지 확인
        total_allocation = sum(alloc for _, alloc in self.allocation_strategy)
        assert total_allocation == self.batch_size, f"Allocation mismatch: {total_allocation} != {self.batch_size}"
        
        print(f"📝 Fixed batch calculation:")
        print(f"   Target batch size: {self.batch_size}")
        print(f"   Base allocation per combination: {base_allocation}")
        print(f"   Extra slots distributed: {extra_slots}")
    
    def __iter__(self):
        """Subject별로 클래스가 균등한 배치 생성 - 배치 크기 전략에 따라 분기"""
        if not self.subject_class_indices:
            return
        
        if self.use_dynamic_batch_size:
            yield from self._iter_dynamic_batch()
        else:
            yield from self._iter_fixed_batch()
    
    def _iter_dynamic_batch(self):
        """동적 배치 크기 방식 (기존 로직)"""
        # 최소 샘플 수 확인
        min_samples = min(len(indices) for indices in self.subject_class_indices.values())
        max_batches = min_samples // self.samples_per_subject_class
        
        # 위치 추적 (순환 샘플링용)
        positions = {key: 0 for key in self.subject_class_indices.keys()}
        
        for batch_idx in range(max_batches):
            batch_indices = []
            
            # 사용 가능한 (subject, class) 조합에서만 샘플링
            for key in self.subject_class_indices.keys():
                indices_list = self.subject_class_indices[key]
                current_pos = positions[key]
                
                # 이 조합에서 샘플 선택
                for i in range(self.samples_per_subject_class):
                    idx = (current_pos + i) % len(indices_list)
                    batch_indices.append(indices_list[idx])
                
                # 위치 업데이트
                positions[key] = (current_pos + self.samples_per_subject_class) % len(indices_list)
            
            # 배치 검증 및 반환
            if len(batch_indices) > 0:
                if self.shuffle:
                    random.shuffle(batch_indices)
                yield batch_indices
            else:
                raise ValueError("No valid batch indices available")
    
    def _iter_fixed_batch(self):
        """고정 배치 크기 방식 (새로운 로직)"""
        # 최소 샘플 수 확인 (할당량 기준)
        min_samples_per_combination = {}
        for key, allocation in self.allocation_strategy:
            indices_list = self.subject_class_indices[key]
            min_samples_per_combination[key] = len(indices_list) // allocation
        
        max_batches = min(min_samples_per_combination.values()) if min_samples_per_combination else 0
        
        # 위치 추적 (순환 샘플링용)
        positions = {key: 0 for key in self.subject_class_indices.keys()}
        
        for batch_idx in range(max_batches):
            batch_indices = []
            
            # 할당 전략에 따라 샘플링
            for key, allocation in self.allocation_strategy:
                indices_list = self.subject_class_indices[key]
                current_pos = positions[key]
                
                # 이 조합에서 할당된 만큼 샘플 선택
                for i in range(allocation):
                    idx = (current_pos + i) % len(indices_list)
                    batch_indices.append(indices_list[idx])
                
                # 위치 업데이트
                positions[key] = (current_pos + allocation) % len(indices_list)
            
            # 배치 크기 검증
            assert len(batch_indices) == self.batch_size, f"Batch size mismatch: {len(batch_indices)} != {self.batch_size}"
            
            # 배치 반환
            if self.shuffle:
                random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        """총 배치 수 반환 - 배치 크기 전략에 따라 분기"""
        if not self.subject_class_indices:
            return 0
        
        if self.use_dynamic_batch_size:
            min_samples = min(len(indices) for indices in self.subject_class_indices.values())
            return min_samples // self.samples_per_subject_class
        else:
            # 고정 배치 크기: 할당량 기준으로 계산
            min_batches_per_combination = []
            for key, allocation in self.allocation_strategy:
                indices_list = self.subject_class_indices[key]
                max_batches_for_key = len(indices_list) // allocation
                min_batches_per_combination.append(max_batches_for_key)
            
            return min(min_batches_per_combination) if min_batches_per_combination else 0
    
    def verify_batch_balance(self, batch_indices):
        """배치의 subject별 클래스 분포 검증 (디버깅용)"""
        if not batch_indices:
            return
        
        # 배치에서 실제 데이터 가져오기
        batch_labels = {}
        for idx in batch_indices:
            _, _, label_tensor, _ = self.dataset[idx]
            for subject_idx in range(self.num_subjects):
                if subject_idx < len(label_tensor):
                    label = label_tensor[subject_idx].item()
                    if subject_idx not in batch_labels:
                        batch_labels[subject_idx] = []
                    batch_labels[subject_idx].append(label)
        
        # 분포 출력
        print("📊 Batch balance verification:")
        for subject_idx, labels in batch_labels.items():
            from collections import Counter
            distribution = Counter(labels)
            print(f"   Subject {subject_idx}: {dict(distribution)}")
        

class DMMRCompatibleDataset(EEGDataset):
    """
    DMMR-compatible extension of EEGDataset.
    Handles DMMR-specific batch formatting and subject information.
    """
    
    def __init__(self, data_x, data_y, domain_y=None, subject_y=None, kernels=1, 
                 dmmr_mode=False, time_steps=15, stage='fit', step: str = None):
        super().__init__(data_x, data_y, domain_y, kernels)
        self.subject_y = subject_y
        self.dmmr_mode = dmmr_mode
        self.time_steps = time_steps
        self.stage = stage  # 'fit', 'test', 'val'
        self.step = step # None, 'pretraining', 'finetuning'
        print(f"🔍 DMMR Dataset initialized: dmmr_mode={self.dmmr_mode}, time_steps={self.time_steps}, stage={self.stage}, step={self.step}")
        print(f"🔍 Dataset size: {len(self.data_x)} samples")
        # x=(1728, 78), y=(1728,), domain=(1728,)
        print(f"🔍 Data shapes: x={self.data_x.shape}, y={self.data_y.shape if self.data_y is not None else 'N/A'}, domain={self.domain_y.shape if self.domain_y is not None else 'N/A'}")

    def _get_data(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Get data sample by index.
        
        Args:
            idx: Index of the sample to retrieve.
        
        Returns:
            Tuple of (data, label) where data is a tensor and label is the corresponding class.
        """
        x = self.data_x[idx]
        y = self.data_y[idx]
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __getitem__(self, idx):
        """데이터 아이템 반환 - (data_num, subject_num, time_step, features) 형태 고려"""
        x, y = self._get_data(idx)
        domain_y = self.domain_y[idx] if self.domain_y is not None else 0
        subject_y = self.subject_y[idx] if self.subject_y is not None else 0

        # x는 이미 (subject_num, time_step, features) 형태로 전달됨
        # Convert to tensors, handling both numpy arrays and existing tensors
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = y.long()
            
        if not isinstance(domain_y, torch.Tensor):
            domain_y = torch.tensor(domain_y, dtype=torch.long)
        else:
            domain_y = domain_y.long()
            
        if not isinstance(subject_y, torch.Tensor):
            subject_y = torch.tensor(subject_y, dtype=torch.long)
        else:
            subject_y = subject_y.long()
        
        return x, domain_y, y, subject_y

    @staticmethod
    def subject_dmmr_collate_fn(batch):
        """
        DMMR correspondence 생성 - 원본 DMMR 논문 방식 정확히 구현
        
        원본 구조: 각 subject별로 독립적인 배치 처리 시,
        현재 subject의 라벨에 대해 모든 subject에서 같은 라벨 데이터 선택
        """
        # 입력: batch에서 각 아이템은 (subject_num, time_steps, features) 형태
        batch_data = torch.stack([item[0] for item in batch])  # (batch_size, subject_num, time_steps, features)
        batch_subject_labels = torch.stack([item[1] for item in batch])  # (batch_size, subject_num)
        batch_labels = torch.stack([item[2] for item in batch])  # (batch_size, subject_num)
        
        batch_size, subject_num, time_steps, features = batch_data.shape
        
        # 🎯 1단계: 각 피험자별 현재 배치에서 라벨별 데이터 수집 (원본의 data_dict, label_dict)
        data_dict = {}  # {subject_idx: [data_samples]}
        label_dict = {}  # {subject_idx: [labels]}
        
        for subject_idx in range(subject_num):
            data_dict[subject_idx] = []
            label_dict[subject_idx] = []
            for batch_idx in range(batch_size):
                data_sample = batch_data[batch_idx, subject_idx]  # (time_steps, features)
                label = batch_labels[batch_idx, subject_idx].item()
                data_dict[subject_idx].append(data_sample)
                label_dict[subject_idx].append(label)
        
        # 🎯 2단계: 각 피험자별로 라벨별 데이터 매핑 생성 (원본의 label_data_dict_list)
        label_data_dict_list = []
        for subject_idx in range(subject_num):
            label_data_dict = defaultdict(set)  # 원본과 동일하게 set 사용
            cur_data_list = data_dict[subject_idx]
            cur_label_list = label_dict[subject_idx]
            
            for i in range(batch_size):
                label_data_dict[cur_label_list[i]].add(cur_data_list[i])
            
            label_data_dict_list.append(label_data_dict)
        try:
            # 🎯 3단계: (subject_num, batch_size, subject_num, time_steps, features) 형태 생성
            # 각 current_subject에 대해 (batch_size, subject_num, time_steps, features) correspondence 생성
            all_correspondence_data = []
            
            for current_subject in range(subject_num):
                # 현재 subject의 라벨들에 대해
                current_subject_labels = label_dict[current_subject]
                
                # 현재 subject를 위한 correspondence: (batch_size, subject_num, time_steps, features)
                subject_correspondence = []
                
                # 각 배치 샘플에 대해
                for batch_idx in range(batch_size):
                    batch_correspondence = []
                    
                    # 모든 subject에서 correspondence 선택
                    for source_subject_idx in range(subject_num):
                        # 현재 배치 샘플의 라벨 사용
                        target_label = current_subject_labels[batch_idx]
                        
                        # source_subject_idx에서 target_label과 같은 라벨의 데이터 선택
                        selected_data = random.choice(list(label_data_dict_list[source_subject_idx][target_label]))
                        batch_correspondence.append(selected_data)
                    
                    # Stack this batch's correspondences: (subject_num, time_steps, features)
                    subject_correspondence.append(torch.stack(batch_correspondence))
                
                # Stack all batches for this subject: (batch_size, subject_num, time_steps, features)
                all_correspondence_data.append(torch.stack(subject_correspondence))
        except Exception as e:
            print(f"Error during correspondence creation: {e}")
            # label_data_dict_list shape 출력
            for i, ld in enumerate(label_data_dict_list):
                print(f"label_data_dict_list[{i}] shape: {len(ld)}")
                # 각 subject별 데이터 수집 상태 출력
                for label, data_set in ld.items():
                    print(f"  Label {label}: {len(data_set)} samples")
            
            raise ValueError("Failed to create correspondence data due to inconsistent label data.")
        
        # 🎯 4단계: 최종 형태 생성 (subject_num, batch_size * subject_num, time_steps, features)
        # Stack all subjects: (subject_num, batch_size, subject_num, time_steps, features)
        correspondence_data = torch.stack(all_correspondence_data)
        # Reshape to (subject_num, batch_size * subject_num, time_steps, features)
        subject_num_final, batch_size_final, subject_num_inner, time_steps_final, features_final = correspondence_data.shape
        correspondence_data = correspondence_data.view(subject_num_final, batch_size_final * subject_num_inner, time_steps_final, features_final)
        
        # 반환값 구조
        source_data = batch_data  # 원본 형태 그대로 유지
        source_labels = batch_labels  # 원본 형태 그대로 유지
        subject_ids = batch_subject_labels  # 원본 형태 그대로 유지
        subject_mask_data = torch.ones((batch_size, subject_num), dtype=torch.float32)  # 모든 subject에 대해 mask 1로 초기화
        
        return source_data, correspondence_data, subject_ids, source_labels, subject_mask_data

    @staticmethod
    def subject_dmmr_single_label_collate_fn(batch):
        """
        DMMR correspondence 생성 - 원본 DMMR 논문 방식 정확히 구현
        
        원본 구조: 각 subject별로 독립적인 배치 처리 시,
        현재 subject의 라벨에 대해 모든 subject에서 같은 라벨 데이터 선택

        UI, UNM DMMR 실험에서 사용되는 subject별로 단일 라벨만 존재하는 데이터셋에대한 대응 함수
        """
        # 입력: batch에서 각 아이템은 (subject_num, time_steps, features) 형태
        batch_data = torch.stack([item[0] for item in batch])  # (batch_size, subject_num, time_steps, features)
        batch_subject_labels = torch.stack([item[1] for item in batch])  # (batch_size, subject_num)
        batch_labels = torch.stack([item[2] for item in batch])  # (batch_size, subject_num)
        
        batch_size, subject_num, time_steps, features = batch_data.shape
        
        # 🎯 1단계: 각 피험자별 현재 배치에서 라벨별 데이터 수집 (원본의 data_dict, label_dict)
        data_dict = {}  # {subject_idx: [data_samples]}
        label_dict = {}  # {subject_idx: [labels]}
        
        # 각 subject별로 하나의 label만을 가지고 있음.
        for subject_idx in range(subject_num):
            data_dict[subject_idx] = []
            label_dict[subject_idx] = []
            for batch_idx in range(batch_size):
                data_sample = batch_data[batch_idx, subject_idx]  # (time_steps, features)
                label = batch_labels[batch_idx, subject_idx].item()
                data_dict[subject_idx].append(data_sample)
                label_dict[subject_idx].append(label)
        
        # 🎯 2단계: 각 피험자별로 라벨별 데이터 매핑 생성 (원본의 label_data_dict_list)
        label_data_dict_list = []
        for subject_idx in range(subject_num):
            label_data_dict = defaultdict(set)  # 원본과 동일하게 set 사용
            cur_data_list = data_dict[subject_idx]
            cur_label_list = label_dict[subject_idx]
            
            for i in range(batch_size):
                label_data_dict[cur_label_list[i]].add(cur_data_list[i])
            
            label_data_dict_list.append(label_data_dict)
        try:
            # 🎯 3단계: (subject_num, batch_size, subject_num, time_steps, features) 형태 생성
            # 각 current_subject에 대해 (batch_size, subject_num, time_steps, features) correspondence 생성
            all_correspondence_data = []
            # 각 current_subject에 대해 (batch_size, subject_num) mask 생성
            # current_subject 와 동일한 label을 가진 데이터만 모델의 학습에 참여 시키기 위한 mask
            all_subject_mask = []  # 각 subject별 mask 정보
            
            for current_subject in range(subject_num):
                # 현재 subject의 라벨들에 대해
                current_subject_labels = label_dict[current_subject]
                
                # 현재 subject를 위한 correspondence: (batch_size, subject_num, time_steps, features)
                subject_correspondence = []
                subject_subject_mask = []  # 현재 subject의 mask 정보

                # 각 배치 샘플에 대해
                for batch_idx in range(batch_size):
                    batch_correspondence = []
                    batch_subject_mask = []  # 현재 배치의 subject별 mask
                    
                    # 모든 subject에서 correspondence 선택
                    for source_subject_idx in range(subject_num):
                        # 현재 배치 샘플의 라벨 사용
                        target_label = current_subject_labels[batch_idx]
                        mask_value = 0  # 해당 subject와 동일한 label이 없는 경우 mask는 0
                        
                        if target_label not in label_data_dict_list[source_subject_idx].keys():
                            selected_data = torch.zeros((time_steps, features), dtype=torch.float32)  # 해당 라벨이 0으로 채움
                        else:
                            # source_subject_idx에서 target_label과 같은 라벨의 데이터 선택
                            selected_data = random.choice(list(label_data_dict_list[source_subject_idx][target_label]))
                            mask_value = 1  # 현재 subject와 동일한 label을 가진 데이터가 있는 경우 mask는 1
                        
                        batch_subject_mask.append(mask_value)  # 현재 subject와 동일한 label을 가진 데이터
                        batch_correspondence.append(selected_data)
                        
                    
                    # Stack this batch's correspondences: (subject_num, time_steps, features)
                    subject_correspondence.append(torch.stack(batch_correspondence))
                    subject_subject_mask.append(torch.tensor(batch_subject_mask, dtype=torch.float32))  # 현재 배치의 subject별 mask 정보
                
                # Stack all batches for this subject: (batch_size, subject_num, time_steps, features)
                all_correspondence_data.append(torch.stack(subject_correspondence))
                all_subject_mask.append(torch.stack(subject_subject_mask))  # 현재 subject의 mask 정보
        except Exception as e:
            print(f"Error during correspondence creation: {e}")
            # label_data_dict_list shape 출력
            # for i, ld in enumerate(label_data_dict_list):
            #     print(f"label_data_dict_list[{i}] shape: {len(ld)}")
            #     # 각 subject별 데이터 수집 상태 출력
            #     for label, data_set in ld.items():
            #         print(f"  Label {label}: {len(data_set)} samples")
            
            raise ValueError("Failed to create correspondence data due to inconsistent label data.")
        
        # 🎯 4단계: 최종 형태 생성 (subject_num, batch_size * subject_num, time_steps, features)
        # Stack all subjects: (subject_num, batch_size, subject_num, time_steps, features)
        correspondence_data = torch.stack(all_correspondence_data)
        subject_mask_data = torch.stack(all_subject_mask)  # (subject_num, batch_size, subject_num)
        # Reshape to (subject_num, batch_size * subject_num, time_steps, features)
        subject_num_final, batch_size_final, subject_num_inner, time_steps_final, features_final = correspondence_data.shape
        correspondence_data = correspondence_data.view(subject_num_final, batch_size_final * subject_num_inner, time_steps_final, features_final)

        print(f"🔍 Correspondence data shape: {correspondence_data.shape}")
        print(f"🔍 Subject mask data shape: {subject_mask_data.shape}")

        # 반환값 구조
        source_data = batch_data  # 원본 형태 그대로 유지
        source_labels = batch_labels  # 원본 형태 그대로 유지
        subject_ids = batch_subject_labels  # 원본 형태 그대로 유지

        return source_data, correspondence_data, subject_ids, source_labels, subject_mask_data

    @staticmethod
    def dmmr_collate_fn(batch):
        """
        Legacy DMMR collate function - 호환성을 위해 유지
        새로운 Subject별 처리는 subject_dmmr_collate_fn 사용
        """
        return DMMRCompatibleDataset.subject_dmmr_collate_fn(batch)

    @staticmethod
    def dmmr_single_label_collate_fn(batch):
        """
        Legacy DMMR collate function - 호환성을 위해 유지
        새로운 Subject별 처리는 subject_dmmr_single_label_collate_fn 사용
        """
        return DMMRCompatibleDataset.subject_dmmr_single_label_collate_fn(batch)
    
    @staticmethod
    def get_dmmr_collate_fn(is_single_label: bool = False) -> callable:
        """
        DMMR-specific collate function.
        
        Args:
            is_single_label: If True, uses single-label correspondence logic.
                             If False, uses multi-label correspondence logic.
                             
        Returns:
            Correspondence collate function based on label type.
        """
        if is_single_label:
            return DMMRCompatibleDataset.dmmr_single_label_collate_fn
        else:
            return DMMRCompatibleDataset.dmmr_collate_fn


class DMMRCompatibleDataModule(EEGDataModule):
    """
    Zero-disruption DMMR-compatible extension of EEGDataModule.
    
    Adds DMMR functionality while maintaining complete backward compatibility.
    All existing functionality works exactly as before.
    """
    
    def __init__(self, dmmr_mode: bool = True, time_steps: int = 6,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 **kwargs):
        """
        Initialize DMMR-compatible DataModule.
        
        Args:
            dmmr_mode: Enable DMMR-specific functionality
            time_steps: Time steps for DMMR sliding window
            train_ratio: Ratio for training data split (per subject)
            val_ratio: Ratio for validation data split (per subject)  
            test_ratio: Ratio for test data split (per subject)
            **kwargs: All arguments passed to parent EEGDataModule
        """
        self.dmmr_mode = dmmr_mode
        self.time_steps = time_steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        if self.dmmr_mode:
            print("🚀 Initializing DMMR mode with independent subject splitting...")
            
            # Initialize only the base Lightning DataModule (skip EEGdataModuel processing)
            pl.LightningDataModule.__init__(self)
            
            # Manually set essential attributes from kwargs
            self.data_config = kwargs.get('data_config', {})
            self.batch_size = kwargs.get('batch_size', 16)
            self.masking_ch_list = kwargs.get('masking_ch_list', [])
            self.rm_ch_list = kwargs.get('rm_ch_list', [])
            self.subject_usage = kwargs.get('subject_usage', 'all')
            self.seed = kwargs.get('seed', 42)
            self.default_path = kwargs.get('default_path', "/root/workspace/Fairness_for_generalization")
            self.data_augmentation_config = kwargs.get('data_augmentation_config', {})
            self.skip_time_list = kwargs.get('skip_time_list', None)
            
            # Initialize augmentation pipeline
            self.data_augmentation_pipeline = None
            print("🔆" * 20)
            print(f"self.data_augmentation_config: {self.data_augmentation_config}")
            print(f"self.enabled: {self.data_augmentation_config.get('enabled', False)}")
            print("🔆" * 20)
            if self.data_augmentation_config.get('enabled', False):
                self._setup_augmentation()
            
            # Store method configurations for later instantiation
            self._augmentation_method_configs = []
            
            # DMMR-specific attributes
            self.subject_mapping = {}  # {subject_id: index}
            self.subject_counts = {}   # {subject_id: sample_count}
            self.input_dim = None
            self.time_steps = time_steps
            
            # DMMR-specific data processing
            self.data_x, self.data_y, self.domain_y = self.load_and_prepare_dataDict()
            self._extract_subject_information()
            self.get_info_from_data()  # Get data info including shape and classes
            self._determine_dmmr_dimensions()
            
            print("✅ DMMR initialization complete")
            
        else:
            # Use standard EEGDataModule initialization for backward compatibility
            super().__init__(**kwargs)
            
            # DMMR-specific attributes (set defaults for compatibility)
            self.subject_mapping = {}
            self.subject_counts = {}
            self.input_dim = None
            self.time_steps = time_steps
    
    def _apply_sliding_window(self, data, time_steps=15):
        """
        Apply sliding window processing identical to original DMMR's window_slice.
        
        Args:
            data: Input data with shape (trials, channels, features) or (samples, features)
            time_steps: Window size (default: 15)
            
        Returns:
            Windowed data with shape (num_windows, time_steps, input_dim)
        """
        # Handle different input shapes
        if len(data.shape) == 3:
            # 3D input: (trials, channels, features) → reshape to (samples, features)
            data = np.transpose(data, (1, 0, 2)).reshape(-1, data.shape[1] * data.shape[2])
        
        # Apply sliding window
        windows = []
        for i in range(data.shape[0] - time_steps + 1):
            windows.append(data[i:i + time_steps])
        return np.array(windows)  # Shape: (num_windows, time_steps, features)
    
    def _normalize_minmax(self, features):
        """
        Apply Min-Max normalization identical to original DMMR's normalize function.
        
        Args:
            features: Input features to normalize
            
        Returns:
            Normalized features in range [0, 1]
        """
        if len(features) == 0:
            return features
            
        # Convert to numpy if tensor
        if hasattr(features, 'numpy'):
            features = features.numpy()
        elif not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Calculate min and max along the first dimension (samples)
        features_min = np.min(features, axis=0, keepdims=True)
        features_max = np.max(features, axis=0, keepdims=True)

        # Avoid division by zero
        denominator = features_max - features_min
        normalized = (features - features_min) / denominator
        return normalized
    
    def get_info_from_data(self):
        """
        Extracts metadata from the loaded data, such as shape, number of classes,
        and creates label-to-index mappings.
        """
        # Get data shape information from the training set.
        # trials, channel*features(frequency_bands)
        self.data_shape = self.data_x['train'].shape
        self.trails = self.data_shape[0]
        self.total_features_number = self.data_shape[-1]

        # Find all unique labels and domain labels across all splits.
        self.unique_labels = set()
        if self.domain_y is not None:
            self.domain_unique_labels = set()
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            self.unique_labels.update(np.unique(self.data_y[split].flatten()))
            if self.domain_y is not None and self.domain_y[split] is not None and len(self.domain_y[split]) > 0:
                self.domain_unique_labels.update(np.unique(self.domain_y[split].flatten()))

        # Create sorted lists and mappings from labels to integer indices.
        self.unique_labels = sorted(list(self.unique_labels))
        self.num_classes = len(self.unique_labels)
        
        # Create augmentation pipeline now that we have channel information
        if self.data_augmentation_config.get('enabled', False):
            print("🔆 Setting up data augmentation pipeline...")
            self.data_augmentation_pipeline = self._create_augmentation_pipeline()
            if self.data_augmentation_pipeline is None:
                print("Warning: Failed to create augmentation pipeline. Augmentation will be disabled.")

    def load_and_prepare_dataDict(self):
        """
        새로운 방식: 피험자별 독립적인 train/val/test 분할 후 통합
        """
        print("🚀 Loading data with subject-wise independent splitting...")
        
        # 1단계: 피험자별 데이터 수집
        subject_data_dict = self._collect_subject_data()
        
        if not subject_data_dict:
            raise ValueError("No subject data found. Check data configuration.")
        
        subject_data_dict = self._downsample_subject_data(subject_data_dict)
        
        # 2단계: 통합 데이터 딕셔너리 초기화
        data_x = {'train': [], 'val': [], 'test': []}
        data_y = {'train': [], 'val': [], 'test': []}
        domain_y = {'train': [], 'val': [], 'test': []} if 'domain_list' in self.data_config else None
        
        # 3단계: 각 피험자별로 독립 분할 후 피험자별로 저장
        for subject_id, subject_data in subject_data_dict.items():
            print(f"📊 Processing subject {subject_id}: {len(subject_data['data'])} samples")
            
            # 피험자별 독립 분할
            split_result = self._split_subject_data(
                subject_data['data'],
                subject_data['labels'],
                subject_data['domains']
            )
            
            # 분할 결과를 피험자별로 구분해서 저장
            for split in ['train', 'val', 'test']:
                if split_result[split]['data'].size > 0:
                    # 피험자별로 데이터 저장
                    data_x[split].append(np.array(split_result[split]['data']))
                    data_y[split].append(np.array(split_result[split]['labels']))

                    if domain_y is not None:
                        domain_y[split].append(np.array(split_result[split]['domains']))
                else:
                    raise ValueError(f"No data found for subject {subject_id} in split {split}. Check your data configuration.")


                print(f"📈 {subject_id} {split} split: {len(data_x[split][-1])} samples")
                print(f"📈 {subject_id} {split} split: {len(data_x[split])}")


        # 4단계: data_num, time_step, features => data_num, subject, time_step, features 형태로 변경
        for split in ['train', 'val', 'test']:
            data_x[split] = np.stack(data_x[split], axis=0).swapaxes(0, 1) 
            data_y[split] = np.stack(data_y[split], axis=0).swapaxes(0, 1)
            if domain_y is not None:
                domain_y[split] = np.stack(domain_y[split], axis=0).swapaxes(0, 1)

            print(f"🔍 Final {split} data shape: {data_x[split].shape}, labels shape: {data_y[split].shape}"
                  f", domain labels shape: {domain_y[split].shape if domain_y is not None else 'N/A'}")
        
        return data_x, data_y, domain_y

    
    def _extract_subject_information(self):
        """
        Extract subject information from file paths in data configuration.
        Supports multiple filename patterns.
        """
        subjects_found = set()
        
        # Pattern 1: {subject}_{condition}_{day}.npy (e.g., B202_before_day1.npy)
        pattern1 = re.compile(r'([A-Z]\d+)_.*\.npy$')
        
        # Pattern 2: {class}_{subject}.npy (e.g., PD_1571.npy, CTL_1411.npy)
        pattern2 = re.compile(r'([A-Z]+)_(\d+)\.npy$')
        
        # Search through all data_list entries
        for split in self.data_config.get('data_list', {}):
            for label in self.data_config['data_list'][split]:
                for file_path in self.data_config['data_list'][split][label]:
                    if not file_path:
                        continue
                    
                    filename = os.path.basename(file_path)

                    # Try pattern 1 (raw3, raw5&6)
                    match = pattern1.match(filename)
                    if match:
                        subject_id = match.group(1)
                        subjects_found.add(subject_id)
                        continue
                    
                    # Try pattern 2 (UI, UNM)
                    match = pattern2.match(filename)
                    if match:
                        subject_id = f"{match.group(1)}_{match.group(2)}"
                        subjects_found.add(subject_id)
                        continue
        
        # Create subject mapping
        self.subject_mapping = {
            subject: idx for idx, subject in enumerate(sorted(subjects_found))
        }
        
        print(f"🔍 Found {len(self.subject_mapping)} subjects: {list(self.subject_mapping.keys())}")
        
        # Subject extraction completed
    
    def _collect_subject_data(self):
        """
        모든 피험자의 데이터를 subject별로 수집
        Returns: Dict[str, Dict[str, List]] - {subject_id: {'data': [...], 'labels': [...], 'domains': [...]}}
        """
        subject_data_dict = {}
        
        print("🔍 Collecting data by subject...")
        
        # Create mappings from file paths to labels and domain labels
        file_to_label = {}
        file_to_domain = {} if 'domain_list' in self.data_config else None

        # Process all splits together to collect by subject
        for split in ['train', 'val', 'test']:
            if split not in self.data_config.get('data_list', {}):
                continue
                
            for label in self.data_config['data_list'][split]:
                for file_path in self.data_config['data_list'][split][label]:
                    full_path = os.path.join(self.default_path, file_path)
                    file_to_label[full_path] = int(label)

            for domain_label in self.data_config['domain_list'][split]:
                for file_path in self.data_config['domain_list'][split][domain_label]:
                    full_path = os.path.join(self.default_path, file_path)
                    if full_path not in file_to_label:
                        raise ValueError(f"File {full_path} not found in domain_list for split {split}.")
                    else:
                        file_to_domain[full_path] = int(domain_label)
        
        # Group files by subject
        for file_path, label in file_to_label.items():
            subject_id = self._extract_subject_from_path(file_path)
            if not subject_id:
                print(f"⚠️  Could not extract subject ID from: {file_path}")
                continue
                
            if subject_id not in subject_data_dict:
                subject_data_dict[subject_id] = {
                    'data': [],
                    'labels': [],
                    'domains': [],
                    'file_paths': []
                }
            
            try:
                # Load and process data
                data = np.load(file_path)
                windowed_data = self._apply_sliding_window(data, self.time_steps)
                normalized_data = self._normalize_minmax(windowed_data)
                
                # Add each window as a separate sample
                for sample in normalized_data:
                    subject_data_dict[subject_id]['data'].append(sample)
                    subject_data_dict[subject_id]['labels'].append(int(label))
                    subject_data_dict[subject_id]['file_paths'].append(file_path)

                    # Add domain label if available
                    if file_to_domain and file_path in file_to_domain:
                        subject_data_dict[subject_id]['domains'].append(file_to_domain[file_path])
                    else:
                        subject_data_dict[subject_id]['domains'].append(None)
                        
                print(f"📁 {subject_id}: Loaded {len(normalized_data)} samples from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
        
        # Convert lists to arrays for each subject
        for subject_id in subject_data_dict:
            subject_data_dict[subject_id]['data'] = np.array(subject_data_dict[subject_id]['data'])
            subject_data_dict[subject_id]['labels'] = np.array(subject_data_dict[subject_id]['labels'])
            if any(d is not None for d in subject_data_dict[subject_id]['domains']):
                subject_data_dict[subject_id]['domains'] = np.array(subject_data_dict[subject_id]['domains'])
            else:
                subject_data_dict[subject_id]['domains'] = None
                
        print(f"✅ Collected data for {len(subject_data_dict)} subjects")
        for subject_id, data in subject_data_dict.items():
            print(f"  📊 {subject_id}: {len(data['data'])} samples")
            
        return subject_data_dict
    
    def analyze_subject_label_distribution(self, data_list):
        """
        각 subject별 가지고 있는 label 종류 분석
        Returns: dict{subject_id: set(labels)}
        """
        subject_labels = defaultdict(set)
        
        for label, file_paths in data_list.items():
            for file_path in file_paths:
                # 파일명에서 subject ID 추출 (PD_1261, CTL_913 등)
                subject_id = self.extract_subject_id(file_path)
                if subject_id:
                    subject_labels[subject_id].add(int(label))
        
        return subject_labels

    def is_single_label_dataset(self, data_list):
        """
        모든 subject가 single label만 가지는지 확인
        Returns: bool
        """
        subject_labels = self.analyze_subject_label_distribution(data_list)
        
        if not subject_labels:
            return False
            
        # 모든 subject가 label 1개씩만 가지면 True
        return all(len(labels) == 1 for labels in subject_labels.values())

    def extract_subject_id(self, file_path):
        """
        파일 경로에서 subject ID 추출
        예: "data/preprocessed/UI_for_DMMR/PD_1261.npy" -> "PD_1261"
        """
        filename = os.path.basename(file_path)
        return filename.split('.')[0]  # 확장자 제거

    def _downsample_subject_data(self, subject_data_dict):
        """
        Downsample subject data to ensure equal number of samples per (subject, class) combination.
        All subjects will have the same number of samples for each class.
        
        Args:
            subject_data_dict: Dictionary of subject data
        
        Returns:
            Downsampled subject data dictionary with balanced (subject, class) combinations
        """
        print("🔍 Downsampling subject data to ensure equal (subject, class) sample counts...")
        
        # 1단계: 모든 subject에서 각 클래스별 최소 샘플 수 찾기
        global_min_samples_per_class = {}
        
        for subject_id, data in subject_data_dict.items():
            from collections import Counter
            label_counts = Counter(data['labels'])
            print(f"📊 {subject_id} label distribution: {dict(label_counts)}")
            
            for label, count in label_counts.items():
                if label not in global_min_samples_per_class:
                    global_min_samples_per_class[label] = count
                else:
                    global_min_samples_per_class[label] = min(global_min_samples_per_class[label], count)
        
        print(f"🎯 Per-class minimum samples: {dict(global_min_samples_per_class)}")
        
        # 전체 클래스들 중에서 절대 최소값을 찾아서 모든 클래스에 적용
        global_absolute_minimum = min(global_min_samples_per_class.values())
        for label in global_min_samples_per_class.keys():
            global_min_samples_per_class[label] = global_absolute_minimum
        
        print(f"🎯 Applied global minimum ({global_absolute_minimum}) to all classes: {dict(global_min_samples_per_class)}")
        
        # 2단계: 모든 subject를 동일한 기준으로 다운샘플링
        downsampled_data = {}
        
        for subject_id, data in subject_data_dict.items():
            print(f"📊 Processing {subject_id}...")
            
            # Collect indices for each label
            label_indices = {}
            for idx, label in enumerate(data['labels']):
                if label not in label_indices:
                    label_indices[label] = []
                label_indices[label].append(idx)
            
            # 각 클래스에서 global minimum 만큼 샘플링
            selected_indices = []
            for label, indices in label_indices.items():
                target_count = global_min_samples_per_class[label]
                if len(indices) < target_count:
                    print(f"⚠️  {subject_id} class {label}: only {len(indices)} samples, need {target_count}")
                    selected_label_indices = indices  # 모든 샘플 사용
                else:
                    # Randomly select target_count samples for this label
                    selected_label_indices = np.random.choice(
                        indices, target_count, replace=False
                    ).tolist()
                selected_indices.extend(selected_label_indices)
            
            # Sort indices to maintain some order
            selected_indices.sort()
            
            # Create downsampled data for this subject
            downsampled_data[subject_id] = {
                'data': data['data'][selected_indices],
                'labels': data['labels'][selected_indices],
                'domains': data['domains'][selected_indices] if data['domains'] is not None else None,
                'file_paths': [data['file_paths'][i] for i in selected_indices]
            }
            
            # Verify label balance
            final_counts = Counter(downsampled_data[subject_id]['labels'])
            print(f"  📉 Downsampled {subject_id}: {len(downsampled_data[subject_id]['data'])} samples")
            print(f"  Final label distribution: {dict(final_counts)}")
        
        return downsampled_data

    
    def _split_subject_data(self, subject_data, subject_labels, subject_domains=None):
        """
        개별 피험자의 데이터를 train/val/test로 분할
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter
        
        if len(subject_data) == 0:
            return {}, {}, {}
            
        # Check if we have enough samples for splitting
        min_samples_per_class = 3  # At least 1 for each split
        label_counts = Counter(subject_labels)
        
        if min(label_counts.values()) < min_samples_per_class:
            print(f"⚠️  Subject has insufficient samples for proper splitting: {dict(label_counts)}")
            # Use simple random split without stratification
            # return self._simple_random_split(subject_data, subject_labels, subject_domains)
            raise NotImplementedError("Insufficient samples for stratified split. Implement fallback logic if needed.")
        else:
            # Use stratified split
            return self._stratified_split(subject_data, subject_labels, subject_domains)
    
    def _simple_random_split(self, subject_data, subject_labels, subject_domains=None):
        """Simple random split without stratification"""
        from sklearn.model_selection import train_test_split
        
        # First split: train + (val + test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            subject_data, subject_labels,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.seed
        )
        
        # Second split: val + test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=(1 - val_size),
            random_state=self.seed
        )
        
        # Handle domains if provided
        train_domains, val_domains, test_domains = None, None, None
        if subject_domains is not None:
            train_domains, temp_domains, _, _ = train_test_split(
                subject_domains, subject_labels,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.seed
            )
            val_domains, test_domains, _, _ = train_test_split(
                temp_domains, temp_labels,
                test_size=(1 - val_size),
                random_state=self.seed
            )
        
        return {
            'train': {'data': train_data, 'labels': train_labels, 'domains': train_domains},
            'val': {'data': val_data, 'labels': val_labels, 'domains': val_domains}, 
            'test': {'data': test_data, 'labels': test_labels, 'domains': test_domains}
        }
    
    def _stratified_split(self, subject_data, subject_labels, subject_domains=None):
        """Stratified split maintaining class balance"""
        from sklearn.model_selection import train_test_split
        
        # First split: train + (val + test) 
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            subject_data, subject_labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=subject_labels,
            random_state=self.seed
        )
        
        # Second split: val + test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            test_size=(1 - val_size), 
            stratify=temp_labels,
            random_state=self.seed
        )
        
        # Handle domains if provided
        train_domains, val_domains, test_domains = None, None, None
        if subject_domains is not None:
            train_domains, temp_domains, _, _ = train_test_split(
                subject_domains, subject_labels,
                test_size=(self.val_ratio + self.test_ratio),
                stratify=subject_labels,
                random_state=self.seed
            )
            val_domains, test_domains, _, _ = train_test_split(
                temp_domains, temp_labels,
                test_size=(1 - val_size),
                stratify=temp_labels, 
                random_state=self.seed
            )
        
        return {
            'train': {'data': train_data, 'labels': train_labels, 'domains': train_domains},
            'val': {'data': val_data, 'labels': val_labels, 'domains': val_domains},
            'test': {'data': test_data, 'labels': test_labels, 'domains': test_domains}
        }
    
    def _determine_dmmr_dimensions(self):
        """Determine input dimensions for DMMR model."""
        print("🔍 Determining DMMR dimensions...")
        
        # Try to get dimensions from loaded data
        try:
            train_data = self.data_x['train']
            if train_data is not None and len(train_data) > 0:
                sample = train_data[0][0]
                if hasattr(sample, 'shape'):
                    if len(sample.shape) == 3:
                        self.time_steps = min(sample.shape[0], 15)
                        self.input_dim = sample.shape[1] * sample.shape[2]
                        # 3D raw data processed
                    elif len(sample.shape) == 2:
                        self.time_steps = min(sample.shape[0], 15)
                        self.input_dim = sample.shape[1]
                        # 2D raw data processed
                    else:
                        raise ValueError(f"Unexpected raw data shape: {sample.shape}")
                    
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            print(f"⚠️  Could not determine dimensions from data: {e}")
            raise ValueError("Failed to determine DMMR dimensions from data. Ensure data is loaded correctly.")
    
    def _extract_subject_from_path(self, file_path: str) -> Optional[str]:
        """Extract subject ID from file path."""
        filename = os.path.basename(file_path)
        
        # Pattern 1: {subject}_{condition}_{day}.npy
        match = re.match(r'^([A-Z]\d+)_.*\.npy$', filename)
        if match:
            return match.group(1)
        
        # Pattern 2: {class}_{subject}.npy
        match = re.match(r'^([A-Z]+)_(\d+)\.npy$', filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}"

        return None
    
    def _create_subject_labels(self, data_paths: List[str]) -> List[int]:
        """Create subject labels for given data paths."""
        subject_labels = []
        
        for path in data_paths:
            subject_id = self._extract_subject_from_path(path)
            if subject_id and subject_id in self.subject_mapping:
                subject_labels.append(self.subject_mapping[subject_id])
            else:
                subject_labels.append(0)  # Default fallback
        
        return subject_labels
    
    @property
    def dmmr_params(self) -> Dict[str, Any]:
        """Get DMMR model parameters."""
        return {
            'number_of_source': len(self.subject_mapping),
            'number_of_category': self.nb_classes if hasattr(self, 'nb_classes') else 2,
            'batch_size': self.batch_size,
            'time_steps': self.time_steps,
            'input_dim': self.input_dim,
            'subject_mapping': self.subject_mapping,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio
        }

    def setup_fit(self, step: Optional[str] = None):
        """Override to add DMMR-specific dataset creation."""
        # First load data using parent implementation
        super().setup_fit()
        
        # Then setup DMMR-specific functionality
        self._setup_dmmr_datasets('fit', step=step)
    
    def setup_test(self, step: Optional[str] = None):
        """Override to add DMMR-specific test dataset creation."""
        super().setup_test()
        # Then setup DMMR-specific functionality
        self._setup_dmmr_datasets('test', step=step)
    
    def _flatten_dmmr_datasets(self):
        # (data_num, subject_num, time_step, features)
        # (data_num, subject_num, time_step, features) => (data_num*subject_num, time_step, features)
        for split in ['train', 'val', 'test']:
            self.data_x[split] = self.data_x[split].reshape(-1, self.time_steps, self.input_dim)
            self.data_y[split] = self.data_y[split].reshape(-1)
            if self.domain_y and split in self.domain_y:
                self.domain_y[split] = self.domain_y[split].reshape(-1)
    
    def _setup_dmmr_datasets(self, stage: str, step: Optional[str] = None):
        """Setup DMMR-compatible datasets with unified approach"""
        # Update dimensions from actual loaded data
        self._determine_dmmr_dimensions()
        if step=='finetuning':
            print("🔧 Fine-tuning mode detected, adjusting time steps and input dimensions...")
            self._flatten_dmmr_datasets()
        
        print(f"🔍 Setting up DMMR datasets for stage '{stage}' with step '{step}'")

        if stage == 'fit':
            print("🔧 Setting up DMMR datasets with unified approach...")
            
            # Subject labels 생성 (각 샘플이 어떤 피험자 것인지)
            train_subject_labels = self._create_subject_labels_for_data(self.data_x['train'], 'train')
            
            self.train_dataset = DMMRCompatibleDataset(
                data_x=self.data_x['train'],
                data_y=self.data_y['train'],
                domain_y=self.domain_y['train'] if self.domain_y else None,
                subject_y=train_subject_labels,
                kernels=1,
                dmmr_mode=True,
                time_steps=self.time_steps,
                stage='fit',
                step=step
            )
            print(f"✅ Train dataset: {len(self.train_dataset)} samples")
        
            val_subject_labels = self._create_subject_labels_for_data(self.data_x['val'], 'val')
            
            self.val_dataset = DMMRCompatibleDataset(
                data_x=self.data_x['val'],
                data_y=self.data_y['val'],
                domain_y=self.domain_y['val'] if self.domain_y else None,
                subject_y=val_subject_labels,
                kernels=1,
                dmmr_mode=True,
                time_steps=self.time_steps,
                stage='val',
                step=step
            )
            print(f"✅ Val dataset: {len(self.val_dataset)} samples")
            
        elif stage == 'test':
            test_subject_labels = self._create_subject_labels_for_data(self.data_x['test'], 'test')
            
            self.test_dataset = DMMRCompatibleDataset(
                data_x=self.data_x['test'],
                data_y=self.data_y['test'],
                domain_y=self.domain_y['test'] if self.domain_y else None,
                subject_y=test_subject_labels,
                kernels=1,
                dmmr_mode=True,
                time_steps=self.time_steps,
                stage='test',
                step=step
            )
            print(f"✅ Test dataset: {len(self.test_dataset)} samples")
    
    def _create_subject_labels_for_data(self, data_samples, split: str):
        """
        통합된 데이터에서 각 샘플의 피험자 라벨을 생성
        
        현재는 모든 피험자가 동일하게 분포되어 있다고 가정하여 순차적으로 할당
        실제로는 데이터 로딩 시점에 추적해야 하나, 간단한 구현으로 우선 진행
        """
        num_subjects = len(self.subject_mapping)
        num_samples = len(data_samples)
        
        # 간단한 구현: 샘플을 피험자 수로 나누어 균등 분배
        # 실제로는 데이터 로딩 시 subject 정보를 추적하는 것이 더 정확함
        subject_labels = []
        samples_per_subject = num_samples // num_subjects
        
        subject_ids = sorted(self.subject_mapping.keys())
        for i, subject_id in enumerate(subject_ids):
            subject_idx = self.subject_mapping[subject_id]
            
            if i == len(subject_ids) - 1:
                # 마지막 피험자는 남은 모든 샘플 할당
                remaining_samples = num_samples - len(subject_labels)
                subject_labels.extend([subject_idx] * remaining_samples)
            else:
                subject_labels.extend([subject_idx] * samples_per_subject)
        
        print(f"🔍 Created subject labels for {split}: {len(subject_labels)} samples across {num_subjects} subjects")
        return subject_labels
    
    def get_dmmr_info(self) -> Dict[str, Any]:
        """Get comprehensive DMMR information for debugging."""
        info = {
            'dmmr_mode': self.dmmr_mode,
            'subject_mapping': self.subject_mapping,
            'time_steps': self.time_steps,
            'input_dim': getattr(self, 'input_dim', 'Unknown'),
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
        }
        
        if hasattr(self, 'data_x'):
            info['data_shapes'] = {
                split: len(data) if data is not None and hasattr(data, '__len__') else 0 
                for split, data in self.data_x.items()
            }
        
        return info

    def _get_collate_fn(self, data_list) -> Optional[Callable]:
        step = self.train_dataset.step
        is_single_label = self.is_single_label_dataset(data_list)
        print(f"🔍 Using collate_fn for step '{step}': Single-label={is_single_label}")
        if step == 'pretraining':
            return DMMRCompatibleDataset.get_dmmr_collate_fn(is_single_label=is_single_label)
        elif step == 'finetuning':
            return None
        else:
            raise ValueError(f"Unknown step '{step}' for collate_fn. Supported: 'pretraining', 'finetuning'.")
    
    def _create_train_dataloader_adaptive(self, train_dataset):
        """
        데이터 특성에 따라 적응적으로 DataLoader 생성
        """
        try:
            # 현재 데이터셋의 label 분포 분석
            data_list = self.data_config['data_list']['train']

            # collate_fn 설정
            collate_fn = self._get_collate_fn(data_list)
            
            if self.is_single_label_dataset(data_list):
                # Case 1: Single-label subjects (UI/UNM 타입)
                print("🎯 Single-label subjects detected → Using Simple DataLoader")
                return self._create_simple_dataloader(train_dataset, collate_fn=collate_fn)
            else:
                # Case 2: Multi-label subjects (raw3/raw5/raw6 타입) 
                print("🎯 Multi-label subjects detected → Using Balanced BatchSampler")
                return self._create_balanced_dataloader(train_dataset, collate_fn=collate_fn)
                
        except Exception as e:
            raise ValueError(f"Failed to create adaptive train DataLoader: {e}")

    def _create_simple_dataloader(self, dataset, collate_fn: Optional[Callable] = None):
        """
        UI/UNM용 단순 DataLoader (balanced sampling 불필요)
        """
        print("📦 Creating Simple DataLoader for single-label subjects")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    def _create_balanced_dataloader(self, dataset, collate_fn: Optional[Callable] = None):
        """
        raw3/raw5/raw6용 SubjectAwareBalancedBatchSampler 사용
        """
        print("⚖️  Creating Balanced DataLoader for multi-label subjects")
        batch_sampler = SubjectAwareBalancedBatchSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            num_classes=self.num_classes if hasattr(self, 'num_classes') else 2,
            num_subjects=len(self.subject_mapping) if self.subject_mapping else 1,
            shuffle=True,
            drop_last=True
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn
        )
    
    def train_dataloader(self, collate_fn: Optional[Callable] = None):
        """통합된 DMMR DataLoader 반환 - 적응적 샘플링 적용"""
        if self.dmmr_mode and hasattr(self, 'train_dataset'):
            print(f"🔄 Returning adaptive DMMR train DataLoader: {len(self.train_dataset)} samples")
            return self._create_train_dataloader_adaptive(self.train_dataset)
        else:
            # 기존 EEG 모드는 그대로
            return super().train_dataloader()
    
    def _create_val_dataloader_adaptive(self, val_dataset):
        """
        Validation용 적응적 DataLoader 생성 (shuffle=False)
        """
        try:
            # 현재 데이터셋의 label 분포 분석
            data_list = self.data_config['data_list']['train']  # train 분포 기준으로 판단

            # collate_fn 설정
            collate_fn = self._get_collate_fn(data_list)
            
            if self.is_single_label_dataset(data_list):
                # Case 1: Single-label subjects (UI/UNM 타입)
                print("🎯 Single-label subjects detected → Using Simple Val DataLoader")
                return DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn
                )
            else:
                # Case 2: Multi-label subjects (raw3/raw5/raw6 타입)
                print("🎯 Multi-label subjects detected → Using Balanced Val BatchSampler")
                batch_sampler = SubjectAwareBalancedBatchSampler(
                    dataset=val_dataset,
                    batch_size=self.batch_size,
                    num_classes=self.num_classes if hasattr(self, 'num_classes') else 2,
                    num_subjects=len(self.subject_mapping) if self.subject_mapping else 1,
                    shuffle=False,
                    drop_last=False
                )
                
                return DataLoader(
                    val_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn
                )
                
        except Exception as e:
            raise ValueError(f"Failed to create adaptive validation DataLoader: {e}")
    
    def val_dataloader(self):
        """PyTorch Lightning이 자동으로 호출하는 validation DataLoader"""
        if self.dmmr_mode and hasattr(self, 'val_dataset'):
            print(f"🔄 Returning adaptive DMMR validation DataLoader: {len(self.val_dataset)} samples")
            return self._create_val_dataloader_adaptive(self.val_dataset)
        else:
            # 기존 EEG 모드는 그대로
            return super().val_dataloader()
    
    def test_dataloader(self):
        """PyTorch Lightning이 자동으로 호출하는 test DataLoader"""
        print(f"🔄 Returning DMMR test DataLoader: {len(self.test_dataset)} samples")
        return self._create_test_dataloader_adaptive(test_dataset=self.test_dataset)

    def _create_test_dataloader_adaptive(self, test_dataset):
        """
        Test용 적응적 DataLoader 생성 (shuffle=False)
        """
        try:
            # 현재 데이터셋의 label 분포 분석
            data_list = self.data_config['data_list']['train']  # train 분포 기준으로 판단

            # collate_fn 설정
            collate_fn = self._get_collate_fn(data_list)

            if self.is_single_label_dataset(data_list):
                # Case 1: Single-label subjects (UI/UNM 타입)
                print("🎯 Single-label subjects detected → Using Simple Test DataLoader")
                return DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            else:
                # Case 2: Multi-label sub`jects (raw3/raw5/raw6 타입)
                print("🎯 Multi-label subjects detected → Using Balanced Test BatchSampler")
                batch_sampler = SubjectAwareBalancedBatchSampler(
                    dataset=test_dataset,
                    batch_size=self.batch_size,
                    num_classes=self.num_classes if hasattr(self, 'num_classes') else 2,
                    num_subjects=len(self.subject_mapping) if self.subject_mapping else 1,
                    shuffle=False,
                    drop_last=False
                )
                
                return DataLoader(
                    test_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn,
                )
                
        except Exception as e:
            raise ValueError(f"Failed to create adaptive test DataLoader: {e}")

# Convenience function for easy DMMR DataModule creation
def create_dmmr_datamodule(config_path: str, **kwargs) -> DMMRCompatibleDataModule:
    """
    Create a DMMR-compatible DataModule with automatic configuration.
    
    Args:
        config_path: Path to data configuration JSON
        **kwargs: Additional arguments for DataModule (train_ratio, val_ratio, test_ratio, etc.)
        
    Returns:
        Configured DMMRCompatibleDataModule
    """
    import json
    
    # Load data configuration
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    
    # Create DataModule with DMMR mode enabled
    datamodule = DMMRCompatibleDataModule(
        data_config=data_config,
        dmmr_mode=True,
        **kwargs
    )
    
    return datamodule