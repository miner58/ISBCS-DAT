import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import math
import random
from collections import defaultdict
from src.data.augmentation.channelwiseDataAugmentation import (
    ChannelwiseDataAugmentation, 
    CorticalRegionChannelSwap, 
    SubjectLevelChannelSwap
)
from src.data.modules.EEGdataModuel import EEGDataModule as BaseEEGDataModule

class EEGPretrainDataset(Dataset):
    """
    DMMR pretrain Dataset that handles multi-source data management and correspondence batch generation.
    
    This Dataset encapsulates the complex batch processing logic from train.py into a clean PyTorch Dataset interface.
    It manages multiple source subjects and generates correspondence batches for DMMR pretraining.
    
    Args:
        data_x (numpy.ndarray): EEG data array (samples × channels × timepoints).
        data_y (numpy.ndarray): Label array (samples,).
        domain_y (numpy.ndarray): Subject/domain ID array (samples,).
        args: Training arguments containing batch_size and source_subjects info.
        current_batch_data: Current iteration's batch data for correspondence generation (optional).
    """
    
    def __init__(self, data_x, data_y, domain_y, args, current_batch_data=None):
        """Initialize the DMMR pretrain dataset with multi-source data."""
        self.args = args
        self.current_batch_data = current_batch_data  # For train.py-style correspondence
        
        # Convert EEGDataModule format to multi-source format
        # data_x: numpy array (samples × channels × timepoints)
        # data_y: numpy array (samples,)  
        # domain_y: numpy array (samples,) - subject IDs
        
        # Organize data by subject
        self.subject_data = {}  # {subject_id: [(data, label), ...]}
        self.label_groups = {}  # {subject_id: {label: [data_indices]}}
        self.all_samples = []  # List of (subject_id, sample_idx_in_subject) tuples
        
        # Get unique subjects and organize data by subject
        unique_subjects = np.unique(domain_y)
        subject_sizes = []  # Track each subject's sample count
        
        for subject_id in unique_subjects:
            # Get samples belonging to this subject
            subject_mask = domain_y == subject_id
            subject_x = data_x[subject_mask]
            subject_y = data_y[subject_mask]
            subject_sizes.append(len(subject_x))
            
            # Store subject data
            self.subject_data[subject_id] = []
            self.label_groups[subject_id] = defaultdict(list)
            
            for i, (data, label) in enumerate(zip(subject_x, subject_y)):
                self.subject_data[subject_id].append((data, label))
                self.label_groups[subject_id][label].append(i)
        
        # Calculate minimum subject size to match train.py iteration behavior
        # train.py uses min(subject_sizes) as the limiting factor for iterations
        min_subject_size = min(subject_sizes)
        print(f"📊 Subject sizes: {subject_sizes}, using min size: {min_subject_size}")
        
        # Limit each subject to min_subject_size samples to match train.py exactly
        for subject_id in unique_subjects:
            # Only include first min_subject_size samples from each subject
            for i in range(min(min_subject_size, len(self.subject_data[subject_id]))):
                self.all_samples.append((subject_id, i))
        
        # Pre-convert label groups to lists for consistent indexing
        for subject_id in self.label_groups:
            for label in self.label_groups[subject_id]:
                self.label_groups[subject_id][label] = list(self.label_groups[subject_id][label])
    
    def set_current_batch_data(self, current_batch_data):
        """
        Set current iteration's batch data for train.py-style correspondence generation.
        
        Args:
            current_batch_data: Dict with structure {subject_id: {label: [data_samples]}}
        """
        self.current_batch_data = current_batch_data
    
    def __len__(self):
        """Return total number of samples across all subjects."""
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        """
        Get a sample without correspondence data.
        
        Note: Correspondence data will be generated at batch level using custom collate_fn
        to exactly match train.py's batch-wise correspondence generation.
        
        Returns:
            tuple: (original_data, subject_id, label)
                - original_data: The original EEG sample
                - subject_id: Subject ID tensor for the original sample  
                - label: Label of the sample
        """
        # Get the original sample
        subject_id, sample_idx = self.all_samples[idx]
        original_data, label = self.subject_data[subject_id][sample_idx]
        
        # Create subject_id tensor (following train.py pattern for batch size)
        # Note: train.py creates subject_id for entire batch, but here we return single sample
        # DataLoader will batch these together later
        subject_id_tensor = torch.tensor(subject_id, dtype=torch.long)
        
        # Convert to tensors (data is already numpy array from EEGDataModule)
        original_data = torch.from_numpy(original_data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return original_data, subject_id_tensor, label_tensor
    
    def _get_correspondence_sample(self, target_label):
        """
        Get a correspondence sample with the same label from current iteration's data.
        
        This implements the correspondence batch generation logic from train.py exactly:
        - Use only current iteration's batch data for correspondence (train.py style)
        - Fallback to full dataset if current_batch_data is not available
        
        Args:
            target_label: The label to match
            
        Returns:
            numpy.ndarray: The correspondence sample data
        """
        # Use current iteration's data if available (train.py style)
        if self.current_batch_data is not None:
            # Collect all samples with target_label from current batch only
            current_label_samples = []
            
            for subject_id in self.current_batch_data:
                if target_label in self.current_batch_data[subject_id]:
                    current_label_samples.extend(self.current_batch_data[subject_id][target_label])
            
            # Use current batch data if available (exactly like train.py)
            if current_label_samples:
                return random.choice(current_label_samples)
        
        # Fallback: use full dataset (original implementation)
        # This happens when current_batch_data is not set
        label_samples = []
        
        for subject_id in self.subject_data:
            if target_label in self.label_groups[subject_id]:
                subject_label_indices = self.label_groups[subject_id][target_label]
                for sample_idx in subject_label_indices:
                    sample_data, _ = self.subject_data[subject_id][sample_idx]
                    label_samples.append(sample_data)
        
        # Randomly choose one correspondence sample (following train.py logic exactly)
        # Note: train.py assumes label_samples is never empty - no exception handling
        return random.choice(label_samples)

    @staticmethod
    def dmmr_collate_fn(batch):
        """
        Custom collate function that exactly replicates train.py's batch correspondence generation.
        
        This function processes a batch of samples and generates correspondence data
        exactly like train.py does - using only the current batch data for correspondence.
        
        Args:
            batch: List of (original_data, subject_id, label) tuples from __getitem__
            
        Returns:
            tuple: (batch_data, batch_correspondence, batch_subject_ids, batch_labels)
        """
        # Separate batch components
        batch_data = torch.stack([item[0] for item in batch])
        batch_subject_ids = torch.stack([item[1] for item in batch])  
        batch_labels = torch.stack([item[2] for item in batch])
        
        # Group current batch data by subject and label (exactly like train.py)
        # This replicates train.py's data_dict and label_dict logic
        current_batch_grouped = defaultdict(lambda: defaultdict(list))
        
        for i, (data, subject_id, label) in enumerate(batch):
            subject_id_val = subject_id.item()
            label_val = label.item()
            current_batch_grouped[subject_id_val][label_val].append(data)
        
        # Generate correspondence batch (exactly like train.py's corres_batch_data logic)
        correspondence_batch = []
        
        for i, (data, subject_id, label) in enumerate(batch):
            label_val = label.item()
            
            # Collect all samples with same label from current batch (all subjects)
            label_samples = []
            for subj_id in current_batch_grouped:
                if label_val in current_batch_grouped[subj_id]:
                    label_samples.extend(current_batch_grouped[subj_id][label_val])
            
            # Randomly choose correspondence sample (exactly like train.py)
            if label_samples:
                correspondence_sample = random.choice(label_samples)
            else:
                # Fallback: use original sample if no correspondence found
                correspondence_sample = data
                
            correspondence_batch.append(correspondence_sample)
        
        correspondence_batch = torch.stack(correspondence_batch)
        
        return batch_data, correspondence_batch, batch_subject_ids, batch_labels


class EEGFinetuneDataset(Dataset):
    """
    DMMR finetune Dataset for standard emotion classification.
    
    This Dataset provides a simple interface for the DMMR finetuning stage, which performs
    standard supervised learning without complex correspondence batch generation.
    
    Args:
        data_x (numpy.ndarray): EEG data array (samples × channels × timepoints).
        data_y (numpy.ndarray): Label array (samples,).
        domain_y (numpy.ndarray): Domain/subject ID array (samples,) - used for compatibility.
        args: Training arguments (optional, for consistency with other Dataset classes).
    """
    
    def __init__(self, data_x, data_y, domain_y, args):
        """Initialize the DMMR pretrain dataset with multi-source data."""
        self.args = args
        
        # Convert EEGDataModule format to multi-source format
        # data_x: numpy array (samples × channels × timepoints)
        # data_y: numpy array (samples,)  
        # domain_y: numpy array (samples,) - subject IDs
        
        # Organize data by subject
        self.subject_data = {}  # {subject_id: [(data, label), ...]}
        self.label_groups = {}  # {subject_id: {label: [data_indices]}}
        self.all_samples = []  # List of (subject_id, sample_idx_in_subject) tuples
        
        # Get unique subjects and organize data by subject
        unique_subjects = np.unique(domain_y)
        subject_sizes = []  # Track each subject's sample count
        
        for subject_id in unique_subjects:
            # Get samples belonging to this subject
            subject_mask = domain_y == subject_id
            subject_x = data_x[subject_mask]
            subject_y = data_y[subject_mask]
            subject_sizes.append(len(subject_x))
            
            # Store subject data
            self.subject_data[subject_id] = []
            self.label_groups[subject_id] = defaultdict(list)
            
            for i, (data, label) in enumerate(zip(subject_x, subject_y)):
                self.subject_data[subject_id].append((data, label))
                self.label_groups[subject_id][label].append(i)
        
        # Calculate minimum subject size to match train.py iteration behavior
        # train.py uses min(subject_sizes) as the limiting factor for iterations
        min_subject_size = min(subject_sizes)
        print(f"📊 Subject sizes: {subject_sizes}, using min size: {min_subject_size}")
        
        # Limit each subject to min_subject_size samples to match train.py exactly
        for subject_id in unique_subjects:
            # Only include first min_subject_size samples from each subject
            for i in range(min(min_subject_size, len(self.subject_data[subject_id]))):
                self.all_samples.append((subject_id, i))
        
        # Pre-convert label groups to lists for consistent indexing
        for subject_id in self.label_groups:
            for label in self.label_groups[subject_id]:
                self.label_groups[subject_id][label] = list(self.label_groups[subject_id][label])
    
    def __len__(self):
        """Return total number of samples across all subjects."""
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        """
        Get a sample without correspondence data.
        
        Note: Correspondence data will be generated at batch level using custom collate_fn
        to exactly match train.py's batch-wise correspondence generation.
        
        Returns:
            tuple: (original_data, subject_id, label)
                - original_data: The original EEG sample
                - subject_id: Subject ID tensor for the original sample  
                - label: Label of the sample
        """
        # Get the original sample
        subject_id, sample_idx = self.all_samples[idx]
        original_data, label = self.subject_data[subject_id][sample_idx]
        
        # Create subject_id tensor (following train.py pattern for batch size)
        # Note: train.py creates subject_id for entire batch, but here we return single sample
        # DataLoader will batch these together later
        subject_id_tensor = torch.tensor(subject_id, dtype=torch.long)
        
        # Convert to tensors (data is already numpy array from EEGDataModule)
        original_data = torch.from_numpy(original_data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return original_data, subject_id_tensor, label_tensor


class EEGDataset(Dataset):
    """Custom Dataset for EEG data.
    Args:
        data_x (list): List of EEG data samples.
        data_y (list): List of labels corresponding to the EEG data samples.
        domain_y (list, optional): List of domain labels corresponding to the EEG data samples.
        kernels (int, optional): Number of kernels to reshape the data. Default is 1.
    """    
    def __init__(self, data_x, data_y, domain_y=None, kernels=1):
        # Initialize the dataset with data, labels, and optional domain labels.
        self.data_x = data_x
        self.data_y = data_y
        self.domain_y = domain_y
        self.kernels = kernels

    def __len__(self):
        # Return the total number of samples in the dataset.
        return len(self.data_x)

    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset at the given index.
        # print(f"self.data_x.shape: {self.data_x.shape}")
        # print(f"self.data_y.shape: {self.data_y.shape}")
        x = self.data_x[idx]
        y = self.data_y[idx]
        
        # Reshape the data sample to add a new axis for kernels.
        x = x[:, :, np.newaxis]
        x = x.reshape(x.shape[0], x.shape[1], self.kernels)
        
        # Convert the data and labels to PyTorch tensors.
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        
        # Get domain labels if available
        domain_label = None
        if self.domain_y is not None and len(self.domain_y) > 0:
            domain_label = torch.tensor(self.domain_y[idx], dtype=torch.long)
        
        # Return data and labels
        if domain_label is not None:
            return x, (y, domain_label)
        else:
            return x, y

class EEGDataModule(BaseEEGDataModule):
    """
    DMMR-specialized EEG DataModule with subject-wise data processing.
    Inherits from base EEGDataModule and adds DMMR-specific functionality.
    """
    def __init__(self, data_config: dict, batch_size: int = 16, masking_ch_list=None, rm_ch_list=None, 
                 subject_usage: str = "all", seed: int = 42, skip_time_list: dict = None, 
                 default_path: str = "/home/jsw/Fairness/Fairness_for_generalization",
                 data_augmentation_config: dict = None):
        """
        Initialize the DMMR EEGDataModule.
        
        Args:
            data_config (dict): Configuration for data loading.
            batch_size (int): Batch size for DataLoaders.
            masking_ch_list (list, optional): List of channels to mask (zero out).
            rm_ch_list (list, optional): List of channels to remove completely.
            subject_usage (str): How to use subjects ("all" or other options).
            seed (int): Random seed for reproducibility.
            skip_time_list (dict, optional): Time segments to skip in data.
            default_path (str): Default path for data files.
            data_augmentation_config (dict, optional): Configuration for data augmentation.
        """
        # Initialize PyTorch Lightning base class only
        pl.LightningDataModule.__init__(self)
        
        # Store configuration and parameters (same as parent)
        self.data_config = data_config
        self.skip_time_list = skip_time_list
        self.batch_size = batch_size
        self.masking_ch_list = masking_ch_list if masking_ch_list else []
        self.rm_ch_list = rm_ch_list if rm_ch_list else []
        self.subject_usage = subject_usage
        self.seed = seed        
        self.default_path = default_path
        self.data_augmentation_config = data_augmentation_config or {}

        # Initialize augmentation pipeline (using parent method)
        self.data_augmentation_pipeline = None
        print("🔆" * 20)
        print(f"self.data_augmentation_config: {self.data_augmentation_config}")
        print(f"self.enabled: {self.data_augmentation_config.get('enabled', False)}")
        print("🔆" * 20)
        if self.data_augmentation_config.get('enabled', False):
            self._setup_augmentation()
        
        # DMMR-specific data processing
        self.data_x, self.data_y, self.domain_y = self.load_and_prepare_dataDict()  # Parent method
        self.get_info_from_data()  # Parent method
        
        # DMMR-style subject-wise train/val splitting
        self._split_subjects_train_val()

        # self._reshape_data()

    def _split_subjects_train_val(self):
        """Split each subject's data independently into train and validation sets."""
        print("🔄 Splitting each subject independently into train/val...")
        
        if not self.domain_y or 'train' not in self.domain_y or len(self.domain_y['train']) == 0:
            return
            
        # Initialize new data structures
        new_data_x = {'train': [], 'val': [], 'test': self.data_x.get('test', [])}
        new_data_y = {'train': [], 'val': [], 'test': self.data_y.get('test', [])}
        new_domain_y = {'train': [], 'val': [], 'test': self.domain_y.get('test', []) if self.domain_y else []}
        
        # Get unique subjects and process each
        unique_subjects = np.unique(self.domain_y['train'])
        if not any(self.data_config['data_list']['test'].values()):
            for subject_id in unique_subjects:
                subject_mask = self.domain_y['train'] == subject_id
                if np.sum(subject_mask) == 0:
                    continue
                    
                # Extract subject data
                subject_x = self.data_x['train'][subject_mask]
                subject_y = self.data_y['train'][subject_mask]
                subject_d = self.domain_y['train'][subject_mask]
                
                # Split this subject's data: 80% train, 20% val
                s_train_x, s_val_x, s_train_y, s_val_y, s_train_d, s_val_d = train_test_split(
                    subject_x, subject_y, subject_d,
                    test_size=0.2, random_state=self.seed, shuffle=True, stratify=subject_y
                )

                s_val_x, s_test_x, s_val_y, s_test_y, s_val_d, s_test_d = train_test_split(
                    s_val_x, s_val_y, s_train_d,
                    test_size=0.5, random_state=self.seed, shuffle=True, stratify=s_val_y
                )

                # Append to new data structures
                new_data_x['train'].extend(s_train_x)
                new_data_x['val'].extend(s_val_x)
                new_data_y['train'].extend(s_train_y)
                new_data_y['val'].extend(s_val_y)
                new_domain_y['train'].extend(s_train_d)
                new_domain_y['val'].extend(s_val_d)
                new_data_x['test'].extend(s_test_x)
                new_data_y['test'].extend(s_test_y)
                new_domain_y['test'].extend(s_test_d)

            # Convert lists to numpy arrays and update data structures
            for split in ['train', 'val', 'test']:
                self.data_x[split] = np.array(new_data_x[split])
                self.data_y[split] = np.array(new_data_y[split])
                self.domain_y[split] = np.array(new_domain_y[split])

                print(f"  📁 Subject {subject_id}: {len(s_train_x)} train, {len(s_val_x)} val, {len(s_test_x)} test")
        else:
            for subject_id in unique_subjects:
                subject_mask = self.domain_y['train'] == subject_id
                if np.sum(subject_mask) == 0:
                    continue
                    
                # Extract subject data
                subject_x = self.data_x['train'][subject_mask]
                subject_y = self.data_y['train'][subject_mask]
                subject_d = self.domain_y['train'][subject_mask]

                s_train_x, s_val_x, s_train_y, s_val_y, s_train_d, s_val_d = train_test_split(
                    subject_x, subject_y, subject_d,
                    test_size=0.2, random_state=self.seed, shuffle=True, stratify=subject_y
                )

                new_data_x['train'].extend(s_train_x)
                new_data_y['train'].extend(s_train_y)
                new_domain_y['train'].extend(s_train_d)
                new_data_x['val'].extend(s_val_x)
                new_data_y['val'].extend(s_val_y)
                new_domain_y['val'].extend(s_val_d)

            # Convert lists to numpy arrays and update data structures
            for split in ['train', 'val']:
                self.data_x[split] = np.array(new_data_x[split])
                self.data_y[split] = np.array(new_data_y[split])
                self.domain_y[split] = np.array(new_domain_y[split])

    def get_pretrain_dataloader(self, split: str = 'train', args=None) -> DataLoader:
        """
        Get DataLoader for DMMR pretraining stage with correspondence generation.
        
        Args:
            split: Data split ('train', 'val', 'test')
            args: Training arguments needed for Dataset initialization
            
        Returns:
            DataLoader: Configured for DMMR pretraining with custom collate_fn
        """
        if split not in self.data_x or len(self.data_x[split]) == 0:
            raise ValueError(f"No data available for split: {split}")
            
        # Create DMMR pretrain dataset
        dataset = EEGPretrainDataset(
            data_x=self.data_x[split],
            data_y=self.data_y[split], 
            domain_y=self.domain_y[split],
            args=args
        )
        
        # Create DataLoader with custom collate function for correspondence generation
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),  # Shuffle only for training
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with custom collate_fn
            collate_fn=EEGPretrainDataset.dmmr_collate_fn,
            drop_last=True  # Important for consistent batch sizes in DMMR
        )
        
        print(f"📦 Created DMMR Pretrain DataLoader ({split}): {len(dataset)} samples, "
              f"{len(dataloader)} batches")
        
        return dataloader
    
    def get_finetune_dataloader(self, split: str = 'train', args=None) -> DataLoader:
        """
        Get DataLoader for DMMR finetuning stage (standard classification).
        
        Args:
            split: Data split ('train', 'val', 'test')
            args: Training arguments needed for Dataset initialization
            
        Returns:
            DataLoader: Configured for DMMR finetuning (simple classification)
        """
        if split not in self.data_x or len(self.data_x[split]) == 0:
            raise ValueError(f"No data available for split: {split}")
            
        # Create DMMR finetune dataset  
        dataset = EEGFinetuneDataset(
            data_x=self.data_x[split],
            data_y=self.data_y[split],
            domain_y=self.domain_y[split], 
            args=args
        )
        
        # Create standard DataLoader (no custom collate_fn needed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),  # Shuffle only for training
            num_workers=4,  # Can use multiple workers for standard datasets
            drop_last=True  # Consistent with pretraining
        )
        
        print(f"📦 Created DMMR Finetune DataLoader ({split}): {len(dataset)} samples, "
              f"{len(dataloader)} batches") 
        
        return dataloader

    # PyTorch Lightning DataModule interface methods
    def train_dataloader(self, stage: str = 'pretrain', args=None) -> DataLoader:
        """
        PyTorch Lightning interface for training DataLoader.
        
        Args:
            stage: Training stage ('pretrain' or 'finetune')
            args: Training arguments
            
        Returns:
            DataLoader: Configured for the specified training stage
        """
        if stage == 'pretrain':
            return self.get_pretrain_dataloader('train', args)
        elif stage == 'finetune':
            return self.get_finetune_dataloader('train', args)
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'pretrain' or 'finetune'")
    
    def val_dataloader(self, stage: str = 'pretrain', args=None) -> DataLoader:
        """
        PyTorch Lightning interface for validation DataLoader.
        
        Args:
            stage: Training stage ('pretrain' or 'finetune')
            args: Training arguments
            
        Returns:
            DataLoader: Configured for the specified validation stage
        """
        if stage == 'pretrain':
            return self.get_pretrain_dataloader('val', args)
        elif stage == 'finetune':
            return self.get_finetune_dataloader('val', args)
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'pretrain' or 'finetune'")
    
    def test_dataloader(self, stage: str = 'finetune', args=None) -> DataLoader:
        """
        PyTorch Lightning interface for test DataLoader.
        
        Args:
            stage: Training stage ('pretrain' or 'finetune') - typically 'finetune' for testing
            args: Training arguments
            
        Returns:
            DataLoader: Configured for testing
        """
        if stage == 'pretrain':
            return self.get_pretrain_dataloader('test', args)
        elif stage == 'finetune':
            return self.get_finetune_dataloader('test', args)
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'pretrain' or 'finetune'")