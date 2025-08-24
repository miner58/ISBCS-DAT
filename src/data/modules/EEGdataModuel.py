import pytorch_lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import math
from src.data.augmentation.channelwiseDataAugmentation import (
    ChannelwiseDataAugmentation, 
    CorticalRegionChannelSwap, 
    SubjectLevelChannelSwap
)

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
        # print(f"🔆 bef")
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        # print(f"🔆 aft")
        
        # Get domain labels if available
        domain_label = None
        if self.domain_y is not None and len(self.domain_y) > 0:
            domain_label = torch.tensor(self.domain_y[idx], dtype=torch.long)
        
        # print(f"🔆 return")
        # Return data and labels
        if domain_label is not None:
            return x, (y, domain_label)
        else:
            return x, y


class AugmentedCollateFunction:
    """
    Custom collate function that applies batch-wise data augmentation.
    
    This class handles the batching of samples and applies data augmentation
    at the batch level, which is essential for augmentation methods that require
    batch context (e.g., SubjectLevelChannelSwap).
    
    Args:
        augmentation (ChannelwiseDataAugmentation, optional): Augmentation pipeline to apply.
        apply_augmentation (bool): Whether to apply augmentation to this batch.
    """
    
    def __init__(self, augmentation=None, apply_augmentation=True):
        self.augmentation = augmentation
        self.apply_augmentation = apply_augmentation
    
    def return_data(self, x_batch, y_batch, domain_y_batch=None):
        """
        Returns the batched data and labels.
        
        Args:
            x_batch: Batched EEG data samples.
            y_batch: Batched labels.
            domain_y_batch: Batched domain labels (optional).
            
        Returns:
            Tuple of batched tensors: (x_batch, y_batch) or (x_batch, (y_batch, domain_y_batch))
        """
        if domain_y_batch is not None:
            return x_batch, (y_batch, domain_y_batch)
        else:
            return x_batch, y_batch
    
    def __call__(self, batch):
        """
        Collate function that processes a batch of samples.
        
        Args:
            batch: List of samples from the dataset.
            
        Returns:
            Tuple of batched tensors. If augmentation produces soft labels,
            they replace the original domain labels in the output.
        """
        print(f"🔆 Processing batch of size in AugmentedCollateFunction: {len(batch)}")
        # Separate data, labels, and domain labels
        x_batch = torch.stack([item[0] for item in batch])
        y_items = [item[1] for item in batch]
        
        soft_labels = None
        domain_y_batch = None
        
        # Check if domain labels are present
        if isinstance(y_items[0], tuple):
            y_batch, domain_y_batch = zip(*y_items)
            domain_y_batch = torch.stack(domain_y_batch)
            y_batch = torch.stack(y_batch)
        else:
            y_batch = torch.stack(y_items)

        # Apply augmentation if configured
        if self.apply_augmentation and self.augmentation:
            result = self.augmentation(x_batch, y_batch, domain_y_batch)
            
            # Handle both possible return types from augmentation
            if isinstance(result, tuple):
                x_batch, soft_labels = result
            else:
                x_batch = result
        elif self.apply_augmentation and not self.augmentation:
            raise ValueError("Augmentation is enabled but no augmentation pipeline is provided.")
            
        # Prepare final labels tuple. Prioritize soft_labels if they exist.
        if soft_labels is not None:
            return x_batch, (y_batch, soft_labels)
        elif domain_y_batch is not None:
            return x_batch, (y_batch, domain_y_batch)
        else:
            return x_batch, y_batch


class EEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling EEG data.
    This class manages data loading, preprocessing, splitting, and creating DataLoaders.
    """
    def __init__(self, data_config: dict, batch_size: int = 16, masking_ch_list=None, rm_ch_list=None, 
                 subject_usage: str = "all", seed: int = 42, skip_time_list: dict = None, 
                 default_path: str = "/home/jsw/Fairness/Fairness_for_generalization",
                 data_augmentation_config: dict = None):
        """
        Initialize the EEGDataModule.
        
        Args:
            data_config (dict): Configuration for data loading.
            batch_size (int): Batch size for DataLoaders.
            masking_ch_list (list, optional): List of channels to mask (zero out).
            rm_ch_list (list, optional): List of channels to remove completely.
            subject_usage (str): How to use subjects ("all" or other options).
            seed (int): Random seed for reproducibility.
            skip_time_list (dict, optional): Time segments to skip in data.
            default_path (str): Default path for data files.
            augmentation_config (dict, optional): Configuration for data augmentation.
                Expected format:
                {
                    'enabled': bool,  # Whether to enable augmentation
                    'train_only': bool,  # Apply only to training data (default: True)
                    'methods': [  # List of augmentation methods
                        {
                            'type': 'cortical',  # 'cortical' or 'subject'
                            'prob_method': str,  # 'uniform', 'normal', or 'beta'
                            'regions': dict,     # Required for cortical type
                        },
                        # ... more methods
                    ]
                }
        """
        super().__init__()
        # Store configuration and parameters.
        self.data_config = data_config
        self.skip_time_list = skip_time_list
        self.batch_size = batch_size
        self.masking_ch_list = masking_ch_list if masking_ch_list else []
        self.rm_ch_list = rm_ch_list if rm_ch_list else []
        self.subject_usage = subject_usage
        self.seed = seed        
        self.default_path = default_path
        self.data_augmentation_config = data_augmentation_config or {}

        # Initialize augmentation pipeline
        self.data_augmentation_pipeline = None
        print("🔆" * 20)
        print(f"self.data_augmentation_config: {self.data_augmentation_config}")
        print(f"self.enabled: {self.data_augmentation_config.get('enabled', False)}")
        print("🔆" * 20)
        if self.data_augmentation_config.get('enabled', False):
            self._setup_augmentation()
        
        # Prepare data during initialization to avoid repeated work in setup calls.
        self.data_x, self.data_y, self.domain_y = self.load_and_prepare_dataDict()
        self.get_info_from_data()

        

        # Perform dataset splitting during initialization.
        # Check if a test set is explicitly defined in the config.
        if not any(self.data_config['data_list']['test'].values()):
            # If no test data is provided, split the training data into train, validation, and test sets.
            # Split ratio: 80% train, 20% temporary.
            train_files, temp_files, train_labels, temp_labels, train_domain_labels, temp_domain_labels = train_test_split(
                self.data_x['train'],
                self.data_y['train'],
                self.domain_y['train'] if self.domain_y is not None else [None] * len(self.data_y['train']),
                test_size=0.5,
                random_state=self.seed,
                shuffle=True,
                stratify=self.data_y['train'] # Stratify to maintain label distribution.
            )
            # Split the temporary set into validation and test sets (50/50 split).
            # Final split: 80% train, 10% validation, 10% test.
            val_files, test_files, val_labels, test_labels, val_domain_labels, test_domain_labels = train_test_split(
                temp_files,
                temp_labels,
                temp_domain_labels,
                test_size=0.5,
                random_state=self.seed,
                shuffle=True,
                stratify=temp_labels
            )

            # Assign the split data back to the dictionaries.
            self.data_x['train'], self.data_x['val'], self.data_x['test'] = train_files, val_files, test_files
            self.data_y['train'], self.data_y['val'], self.data_y['test'] = train_labels, val_labels, test_labels

            if self.domain_y is not None:
                self.domain_y['train'], self.domain_y['val'], self.domain_y['test'] = train_domain_labels, val_domain_labels, test_domain_labels
        else:
            # If test data is provided, split the training data into train and validation sets.
            # Split ratio: 80% train, 20% validation.
            train_files, val_files, train_labels, val_labels, train_domain_labels, val_domain_labels = train_test_split(
                self.data_x['train'],
                self.data_y['train'],
                self.domain_y['train'] if self.domain_y is not None else [None] * len(self.data_y['train']),
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
                stratify=self.data_y['train']
            )

            # Assign the split data back to the dictionaries.
            self.data_x['train'], self.data_x['val'] = train_files, val_files
            self.data_y['train'], self.data_y['val'] = train_labels, val_labels

            if self.domain_y is not None:
                self.domain_y['train'], self.domain_y['val'] = train_domain_labels, val_domain_labels

    def setup(self, stage: str, step: str = None):
        """
        Assigns train/val/test/predict datasets for use in dataloaders.
        This is a standard PyTorch Lightning method.
        """
        print(f"🔆 Setting up EEGDataModule for stage: {stage}, step: {step}")
        if stage == 'fit' or stage is None:
            self.setup_fit(step)
        if stage == 'validate' or stage is None:
            self.setup_validate(step)
        if stage == 'test' or stage is None:
            self.setup_test(step)
        if stage == 'predict' or stage is None:
            self.setup_predict(step)

    def setup_fit(self, step: str = None):
        """Creates the training and validation datasets."""
        self.train_dataset = EEGDataset(
            data_x=self.data_x['train'],
            data_y=self.data_y['train'],
            domain_y=self.domain_y['train'] if self.domain_y is not None else None,
            kernels=1
        )
        self.val_dataset = EEGDataset(
            data_x=self.data_x['val'],
            data_y=self.data_y['val'],
            domain_y=self.domain_y['val'] if self.domain_y is not None else None,
            kernels=1
        )

    def setup_validate(self, step: str = None):
        """Creates the validation dataset."""
        self.val_dataset = EEGDataset(
            data_x=self.data_x['val'],
            data_y=self.data_y['val'],
            domain_y=self.domain_y['val'] if self.domain_y is not None else None,
            kernels=1
        )

    def setup_test(self, step: str = None):
        """Creates the test dataset."""
        self.test_dataset = EEGDataset(
            data_x=self.data_x['test'],
            data_y=self.data_y['test'],
            domain_y=self.domain_y['test'] if self.domain_y is not None else None,
            kernels=1
        )

    def setup_predict(self, step: str = None):
        """Creates the prediction dataset (using test data)."""
        self.predict_dataset = EEGDataset(
            data_x=self.data_x['test'],
            data_y=self.data_y['test'],
            domain_y=self.domain_y['test'] if self.domain_y is not None else None,
            kernels=1
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set with batch-wise augmentation."""
        # Apply augmentation only to training data by default
        collate_fn = None
        train_augmentation = None
        apply_augmentation = self.data_augmentation_config.get('enabled', False)
        if apply_augmentation:
            train_augmentation = self.data_augmentation_pipeline if self.data_augmentation_config.get('train_only', True) else self.data_augmentation_pipeline
            # Create custom collate function with augmentation
            collate_fn = AugmentedCollateFunction(
                augmentation=train_augmentation,
                apply_augmentation= apply_augmentation
            )
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        """Returns the DataLoader for the prediction set."""
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage: str):
        """Used to clean-up when the run is finished."""
        pass
    
    def load_and_prepare_dataDict(self):
        """
        Loads data from files specified in the config, applies transformations,
        and organizes it into dictionaries.
        """
        # Initialize dictionaries to hold data for each split (train, val, test).
        data_x = {'train': [], 'val': [], 'test': []}
        data_y = {'train': [], 'val': [], 'test': []}
        domain_y = {'train': [], 'val': [], 'test': []} if 'domain_list' in self.data_config else None

        def match_path2label(save_diction: dict, list_name: str) -> None:
            """Helper function to map file paths to their labels."""
            for split in self.data_config[list_name]:
                for label in self.data_config[list_name][split]:
                    for file_path in self.data_config[list_name][split][label]:
                        # The file path acts as a key to identify the object.
                        save_diction[split][os.path.join(self.default_path, file_path)] = label

        def apply_channel_modifications(data) -> np.ndarray:
            """Applies channel masking by zeroing out specified channels."""
            if self.masking_ch_list:
                for n in range(len(data)):
                    for mask_ch in self.masking_ch_list:
                        if mask_ch < data.shape[1]:
                            data[n][mask_ch] = np.zeros_like(data[n][mask_ch])
            return data
        
        def apply_channel_remove(data) -> np.ndarray:
            """Removes specified channels from the data."""
            if self.rm_ch_list:
                data = np.delete(data, self.rm_ch_list, axis=1)
            return data
        
        def apply_skip_time(data, beforeOrAfter:str, subjectName:str) -> np.ndarray:
            """Removes specified time segments from the data."""
            if self.skip_time_list:
                # Get the skip ranges for the specific subject and condition (before/after).
                for skip_range in self.skip_time_list.get(beforeOrAfter, {}).get(subjectName, []):
                    if skip_range:
                        # Delete the specified time range from the data array.
                        # The division by 6 seems specific to the data's time resolution.
                        start_idx = math.floor(skip_range[0] / 6) + 1
                        end_idx = math.ceil(skip_range[1] / 6) + 1
                        data = np.delete(data, np.s_[start_idx:end_idx], axis=0)
                return data
            else:
                return data

        # Create mappings from file paths to labels and domain labels.
        file_to_label = {'train': {}, 'val': {}, 'test': {}}
        file_to_domain = {'train': {}, 'val': {}, 'test': {}} if 'domain_list' in self.data_config else None        
        match_path2label(file_to_label, "data_list")
        if 'domain_list' in self.data_config:
            match_path2label(file_to_domain, "domain_list")
        
        # Process files for each split.
        for split in ['train', 'val', 'test']:
            for file_path, label in file_to_label.get(split, {}).items():
                if file_path is None or label is None:
                    continue
                try:
                    # Load data from the .npy file.
                    data = np.load(file_path)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
                except Exception as e:
                    print(f"Error loading file ({file_path}): {e}")
                    continue

                # Apply data modifications.
                data = apply_channel_modifications(data)
                data = apply_channel_remove(data)

                # Extract attributes from the filename to use in apply_skip_time.
                attrList = file_path.split('_')[-3:]
                subjectName = attrList[0].split('/')[-1]
                beforeOrAfter = attrList[1]
                data = apply_skip_time(data, beforeOrAfter=beforeOrAfter, subjectName=subjectName)

                # Temporary lists to hold data from the current file.
                tmp_data_x = []
                tmp_data_y = []
                tmp_domain_y = []
                # Each item in the loaded 'data' is a separate sample.
                for sample in data:
                    tmp_data_x.append(sample)
                    tmp_data_y.append(label)
                    if 'domain_list' in self.data_config and file_to_domain[split]:
                        domain_label = file_to_domain[split].get(file_path, None)
                        tmp_domain_y.append(domain_label)

                # Extend the main data lists with the data from the current file.
                data_x[split].extend(tmp_data_x)
                data_y[split].extend(tmp_data_y)
                if 'domain_list' in self.data_config and file_to_domain[split]:
                    domain_y[split].extend(tmp_domain_y)

            # Convert the lists of samples to NumPy arrays for efficiency.
            data_x[split] = np.array(data_x[split])
            data_y[split] = np.array(data_y[split])
            if 'domain_list' in self.data_config and file_to_domain[split]:
                domain_y[split] = np.array(domain_y[split])
        
        return data_x, data_y, domain_y

    def get_info_from_data(self):
        """
        Extracts metadata from the loaded data, such as shape, number of classes,
        and creates label-to-index mappings.
        """
        # Get data shape information from the training set.
        print("🔆" * 20)
        print(f"self.data_x['train'].shape: {self.data_x['train'].shape}")
        self.data_shape = self.data_x['train'].shape
        self.chans = self.data_shape[1]
        self.samples = self.data_shape[2]

        # Find all unique labels and domain labels across all splits.
        self.unique_labels = set()
        if self.domain_y is not None:
            self.domain_unique_labels = set()
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            self.unique_labels.update(np.unique(self.data_y[split]))
            if self.domain_y is not None and self.domain_y[split] is not None and len(self.domain_y[split]) > 0:
                self.domain_unique_labels.update(np.unique(self.domain_y[split]))

        # Create sorted lists and mappings from labels to integer indices.
        self.unique_labels = sorted(list(self.unique_labels))
        self.num_classes = len(self.unique_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}

        if self.domain_y is not None:
            self.domain_unique_labels = sorted(list(self.domain_unique_labels))
            self.num_domain_classes = len(self.domain_unique_labels)
            self.domain_label_to_index = {label: idx for idx, label in enumerate(self.domain_unique_labels)}

        # Convert all labels in the data dictionaries to their corresponding integer indices.
        for split in ['train', 'val', 'test']:
            if len(self.data_y[split]) == 0:
                continue
            self.data_y[split] = np.array([self.label_to_index[label] for label in self.data_y[split]])
            if self.domain_y is not None and self.domain_y[split] is not None and len(self.domain_y[split]) > 0:
                self.domain_y[split] = np.array([self.domain_label_to_index[label] for label in self.domain_y[split]])
        
        # Create augmentation pipeline now that we have channel information
        if self.data_augmentation_config.get('enabled', False):
            print("🔆 Setting up data augmentation pipeline...")
            self.data_augmentation_pipeline = self._create_augmentation_pipeline()
            if self.data_augmentation_pipeline is None:
                print("Warning: Failed to create augmentation pipeline. Augmentation will be disabled.")

    def _setup_augmentation(self):
        """
        Setup the data augmentation pipeline based on configuration.
        
        This method creates augmentation methods based on the provided configuration
        and combines them into a ChannelwiseDataAugmentation pipeline.
        """
        try:
            print("🔆 Setting up data augmentation...")
            augmentation_methods = []
            
            for method_config in self.data_augmentation_config.get('methods', []):
                method_type = method_config.get('type', '').lower()
                prob_method = method_config.get('prob_method'  , 'uniform')
                print(f"🔆 Configuring augmentation method: {method_type} with probability method: {prob_method}")
                
                if method_type == 'cortical':
                    # CorticalRegionChannelSwap requires regions and channel count
                    regions = method_config.get('regions', {})
                    print(f"🔆 Regions for cortical augmentation: {regions}")
                    
                    # Handle both dict and list formats for regions
                    if isinstance(regions, dict):
                        # Convert dict to list of lists
                        regions = list(regions.values())
                    elif isinstance(regions, list):
                        # Use list directly
                        regions = regions
                    else:
                        print(f"Warning: 'regions' should be a dict or list. Got {type(regions)}. Skipping.")
                        continue
                    
                    # We'll determine channel count from data after loading
                    # For now, create a placeholder that we'll update later
                    method = {
                        'type': 'cortical',
                        'regions': regions,
                        'prob_method': prob_method
                    }
                    augmentation_methods.append(method)
                    
                elif method_type == 'subject':
                    # SubjectLevelChannelSwap requires channel count
                    # We'll determine this from data after loading
                    print(f"🔆 Subject-level augmentation with probability method: {prob_method}")
                    print(f"🔆 Enable soft labels: {method_config['enable_soft_labels']}")
                    enable_soft_labels = method_config.get('enable_soft_labels', False)
                    method = {
                        'type': 'subject',
                        'prob_method': prob_method,
                        'enable_soft_labels': enable_soft_labels
                    }
                    augmentation_methods.append(method)
                    
                else:
                    print(f"Warning: Unknown augmentation method type '{method_type}'. Skipping.")
            
            # Store method configurations for later instantiation
            self._augmentation_method_configs = augmentation_methods
            
        except Exception as e:
            print(f"Error setting up augmentation: {e}")
            self.augmentation_pipeline = None
    
    def _create_augmentation_pipeline(self):
        """
        Create the actual augmentation pipeline after data is loaded.
        
        This method is called after data loading to create the pipeline with
        the correct channel count information.
        """
        if not hasattr(self, '_augmentation_method_configs') or not self._augmentation_method_configs:
            return None
        
        try:
            print("🔆 Creating augmentation pipeline...")
            augmentation_methods = []
            num_channels = self.chans  # This is available after data loading
            
            for method_config in self._augmentation_method_configs:
                if method_config['type'] == 'cortical':
                    # Handle both dict and list formats for regions
                    regions_data = method_config.get('regions', {})
                    if isinstance(regions_data, dict):
                        # Convert dict to list of lists
                        cortical_regions = list(regions_data.values())
                    elif isinstance(regions_data, list):
                        # Use list directly
                        cortical_regions = regions_data
                    else:
                        print(f"Warning: 'regions' for cortical augmentation should be a dictionary or list. Got {type(regions_data)}. Skipping.")
                        continue
                    
                    method = CorticalRegionChannelSwap(
                        channel_num=num_channels,
                        cortical_regions=cortical_regions,
                        swap_probability_method=method_config['prob_method']
                    )
                    augmentation_methods.append(method)
                    
                elif method_config['type'] == 'subject':
                    enable_soft_labels = method_config.get('enable_soft_labels', False)
                    method = SubjectLevelChannelSwap(
                        channel_num=num_channels,
                        swap_probability_method=method_config['prob_method'],
                        num_domain_classes=self.num_domain_classes,
                        enable_soft_labels=enable_soft_labels
                    )
                    augmentation_methods.append(method)
            
            if augmentation_methods:
                return ChannelwiseDataAugmentation(augmentation_methods)
            else:
                return None
                
        except Exception as e:
            print(f"Error creating augmentation pipeline: {e}")
            return None
