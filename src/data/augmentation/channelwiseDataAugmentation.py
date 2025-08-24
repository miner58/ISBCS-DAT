"""
EEG Channel-wise Data Augmentation Module

This module provides data augmentation techniques specifically designed for EEG (Electroencephalography) 
data. It implements channel swapping strategies that help improve model generalization by simulating 
variations in electrode placement and reducing subject-specific overfitting.

The module includes two main augmentation strategies:
1. CorticalRegionChannelSwap: Swaps channels within anatomically similar cortical regions
2. SubjectLevelChannelSwap: Swaps channel data between different subjects with soft labeling support

These techniques are particularly useful for:
- Cross-subject EEG classification tasks
- Reducing electrode position sensitivity
- Improving model robustness to hardware variations
- Domain adaptation with soft labeling (SubjectLevelChannelSwap)

Author: [Your Name]
Date: 2025-06-29
"""

import torch
import torch.nn.functional as F
import random
from typing import Union, List, Tuple
from collections import defaultdict

# --- Helper Functions for Augmentation ---

def _get_swap_probabilities(num_channels: int, method: str, p=0.5) -> torch.Tensor:
    """
    Generates swap probabilities for each channel using different statistical distributions.
    
    This function creates a probability tensor that determines how likely each channel
    is to participate in a swap operation. Different probability distributions are used
    to create varied augmentation patterns:
    
    - 'normal': Uses absolute normal distribution, good for moderate swap rates
    - 'beta': Uses Beta(0.5, 0.5) distribution, creates U-shaped probability (favors extremes)
    - 'uniform': Uses uniform distribution, provides consistent random swapping
    
    Args:
        num_channels (int): The total number of EEG channels in the dataset.
        method (str): Statistical method for probability generation.
                     Supported: 'normal', 'beta', 'uniform'.

    Returns:
        torch.Tensor: Float tensor of shape (num_channels,) with values in [0, 1].
                     Each element represents the swap probability for that channel.
        
    Raises:
        ValueError: If an unsupported probability method is specified.
        
    Example:
        >>> probs = _get_swap_probabilities(64, 'uniform')
        >>> print(probs.shape)  # torch.Size([64])
        >>> print(torch.all((probs >= 0) & (probs <= 1)))  # True
    """
    if method == 'normal':
        # Generate probabilities from absolute normal distribution
        # abs() ensures non-negative values, % 1.0 maps to [0, 1] range
        return torch.abs(torch.randn(num_channels)) % 1.0
    elif method == 'beta':
        # Beta(0.5, 0.5) creates a U-shaped distribution
        # This favors either very low or very high probabilities
        return torch.distributions.Beta(0.5, 0.5).sample((num_channels,))
    elif method == 'uniform':
        return torch.full((num_channels,), p)  # Fixed probability for uniform distribution
    elif method == 'deterministic':
        # Deterministic probabilities for testing or fixed behavior
        return torch.full((num_channels,), 1.0)  # Fixed probability for uniform distribution
    else:
        raise ValueError(f"Unsupported swap probability method: {method}")


def _decide_swap(swap_probabilities: torch.Tensor) -> torch.Tensor:
    """
    Makes binary swap decisions using Bernoulli trials for each channel.
    
    This function converts continuous probabilities into binary decisions using
    Bernoulli sampling. Each channel is independently evaluated, creating a
    stochastic mask for which channels will participate in swapping.
    
    The Bernoulli distribution is ideal here because:
    - It preserves the intended probability for each channel
    - Decisions are independent across channels
    - Results in natural sparsity (not all channels swap simultaneously)

    Args:
        swap_probabilities (torch.Tensor): Float tensor of probabilities in [0, 1].
                                          Shape should be (num_channels,).

    Returns:
        torch.Tensor: Boolean tensor of the same shape as input.
                     True indicates the channel should be swapped.
                     
    Example:
        >>> probs = torch.tensor([0.1, 0.5, 0.9])
        >>> decisions = _decide_swap(probs)
        >>> print(decisions.dtype)  # torch.bool
        >>> print(decisions.shape)  # torch.Size([3])
    """
    # Bernoulli sampling: each element has its own success probability
    return torch.bernoulli(swap_probabilities).bool()


class SoftLabelGenerator:
    """
    Simple helper class for generating soft domain labels based on channel mixing ratios.
    
    This class converts channel mixing information into probability distributions
    for soft labeling in domain adaptation tasks.
    """
    
    @staticmethod
    def from_channel_counts(channel_counts: torch.Tensor, total_channels: int) -> torch.Tensor:
        """
        Generate soft domain labels from channel source counts.
        
        Args:
            channel_counts (torch.Tensor): Count of channels from each subject.
                                          Shape: (batch_size, num_subjects)
            total_channels (int): Total number of channels.
        
        Returns:
            torch.Tensor: Soft domain labels as probability distribution.
                         Shape: (batch_size, num_subjects)
        
        Example:
            >>> # Sample has 40 channels from subject 0, 24 channels from subject 1 
            >>> counts = torch.tensor([[40, 24]])
            >>> soft_labels = SoftLabelGenerator.from_channel_counts(counts, 64)
            >>> print(soft_labels)  # tensor([[0.625, 0.375]])
        """
        # Convert counts to ratios
        ratios = channel_counts.float() / total_channels
        
        # Normalize to probability distribution (handles edge cases)
        return F.softmax(ratios, dim=1)


# --- Augmentation Modules ---

class CorticalRegionChannelSwap:
    """
    EEG Channel Swapping within Cortical Regions for Data Augmentation.
    
    This class implements an anatomically-informed data augmentation technique that swaps
    EEG channels within predefined cortical regions. The approach is based on the principle
    that channels within the same cortical region capture similar neural activity patterns,
    making their interchange a reasonable augmentation strategy.
    
    Key Benefits:
    - Maintains neurophysiological plausibility by respecting brain anatomy
    - Reduces model sensitivity to exact electrode positions
    - Helps generalize across different EEG cap configurations
    - Preserves spatial relationships within cortical regions
    
    The augmentation process:
    1. For each cortical region, determine which channels will participate in swapping
    2. Randomly shuffle the selected channels within that region
    3. Replace original channel data with shuffled versions
    
    Note: This augmentation is applied per-sample, meaning each sample in a batch
    can have different swap patterns.
    
    Example:
        >>> # Define frontal and parietal regions
        >>> regions = [[0, 1, 2], [3, 4, 5, 6]]  # frontal: 0-2, parietal: 3-6
        >>> augmenter = CorticalRegionChannelSwap(
        ...     channel_num=8, 
        ...     cortical_regions=regions,
        ...     swap_probability_method='uniform'
        ... )
        >>> # Apply to data batch of shape (batch_size, channels, height, width)
        >>> augmented_data = augmenter((data, domain_labels))
    """
    
    def __init__(self, channel_num: int, cortical_regions: List[List[int]], 
                 swap_probability_method: str):
        """
        Initialize the cortical region channel swapper.
        
        Args:
            channel_num (int): Total number of EEG channels in the dataset.
                              Must match the channel dimension of input data.
            cortical_regions (List[List[int]]): Nested list defining cortical regions.
                                               Each inner list contains channel indices
                                               belonging to the same cortical region.
                                               Example: [[0,1,2], [3,4], [5,6,7]]
            swap_probability_method (str): Statistical method for generating swap
                                          probabilities. Options: 'normal', 'beta', 'uniform'.
        
        Raises:
            ValueError: If cortical_regions contains invalid channel indices.
        """
        self.channel_num = channel_num
        self.cortical_regions = cortical_regions
        self.swap_probability_method = swap_probability_method
        
        # Validate cortical regions
        all_channels = [ch for region in cortical_regions for ch in region]
        if any(ch >= channel_num or ch < 0 for ch in all_channels):
            raise ValueError(f"Invalid channel indices in cortical_regions. "
                           f"Must be in range [0, {channel_num-1}]")

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Apply cortical region channel swapping to input data.
        
        This method performs the core augmentation by:
        1. Extracting data tensor from input tuple
        2. Generating swap probabilities for all channels
        3. For each cortical region:
           - Selecting channels to swap based on probabilities
           - Creating a random permutation of selected channels
           - Replacing original data with permuted versions
        
        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): Input tuple containing:
                - data_tensor: EEG data of shape (batch_size, channels, height, width)
                - domain_labels: Subject/domain labels (unused in this method)
        
        Returns:
            torch.Tensor: Augmented data with same shape as input data tensor.
                         Some channels within cortical regions may be swapped.
        
        Note:
            - Swapping only occurs within cortical regions (no cross-region swapping)
            - Regions with fewer than 2 channels are skipped
            - Original data is preserved for channels not selected for swapping
        """
        # Extract data tensor from input tuple (ignore domain labels for this augmentation)
        data_tensor, _, _= data
        augmented_data = data_tensor.clone()
        
        # Generate swap probabilities for all channels
        swap_probabilities = _get_swap_probabilities(self.channel_num, self.swap_probability_method)

        # Process each cortical region independently
        for region in self.cortical_regions:
            # Filter out invalid channel indices (safety check)
            region_channels = [i for i in region if i < self.channel_num]
            
            # Skip regions with insufficient channels for swapping
            if len(region_channels) < 2:
                continue

            # Get swap probabilities for channels in this region
            region_swap_probabilities = swap_probabilities[region_channels]
            
            # Make binary decisions for each channel in the region
            region_swap_decisions = _decide_swap(region_swap_probabilities)
            
            # Collect indices of channels selected for swapping
            swap_indices = [i for i, swap in zip(region_channels, region_swap_decisions) if swap]
            
            # Perform swapping only if at least 2 channels are selected
            if len(swap_indices) >= 2:
                # Create a random permutation of the swap indices
                # This ensures each selected channel gets data from a different channel
                shuffled_indices = random.sample(swap_indices, len(swap_indices))
                
                # Store original data for swapped channels
                original_region_data = augmented_data[:, swap_indices, :, :].clone()
                
                # Apply the permutation: each channel gets data from shuffled position
                for original_idx, shuffled_idx in zip(swap_indices, shuffled_indices):
                    # Find the position of shuffled_idx in the original swap_indices list
                    source_pos = swap_indices.index(shuffled_idx)
                    # Copy data from the source position to the target channel
                    augmented_data[:, original_idx, :, :] = original_region_data[:, source_pos, :, :]
                        
        return augmented_data

# y_label 정보도 받아서 y_label 같은 것들 끼리 처리하도록 수정해야함
class SubjectLevelChannelSwap:
    """
    Cross-Subject EEG Channel Data Swapping for Domain Adaptation.
    
    This augmentation technique swaps channel data between different subjects to improve
    cross-subject generalization in EEG classification tasks. It operates on the principle
    that while individual subjects may have unique neural patterns, the underlying
    signal characteristics for the same cognitive states should be transferable.
    
    Key Applications:
    - Cross-subject EEG classification
    - Domain adaptation in BCI (Brain-Computer Interface) systems
    - Reducing subject-specific overfitting
    - Improving model robustness to inter-subject variability
    
    The Process:
    1. Generate sample-level and channel-level swap probabilities
    2. Group samples by subject and class using pool-based approach
    3. For each selected sample and channel, perform probabilistic swapping
    4. Use dual probability gates for fine-grained control
    
    Important Features:
    - Dual probability control (sample-level + channel-level)
    - Pool-based efficient subject grouping
    - Flexible matching without strict sample count requirements
    - Soft labeling support for domain adaptation
    
    Example:
        >>> # Probabilistic pool-based swapping (default)
        >>> augmenter = SubjectLevelChannelSwap(
        ...     channel_num=64, 
        ...     swap_probability_method='beta',
        ...     sample_probability_method='uniform',
        ...     num_domain_classes=5
        ... )
        >>> augmented_data = augmenter((data, y_labels, domain_labels))
        >>> 
        >>> # Legacy balanced swapping
        >>> augmenter_legacy = SubjectLevelChannelSwap(
        ...     channel_num=64, 
        ...     swap_probability_method='beta',
        ...     num_domain_classes=5,
        ...     use_legacy_method=True
        ... )
        >>> augmented_data = augmenter_legacy((data, y_labels, domain_labels))
    """
    
    def __init__(self, channel_num: int, swap_probability_method: str, num_domain_classes: int, 
                 enable_soft_labels: bool = False, sample_probability_method: str = 'uniform',
                 use_legacy_method: bool = False, debug: bool = False):
        """
        Initialize the subject-level channel swapper.
        
        Args:
            channel_num (int): Total number of EEG channels in the dataset.
            swap_probability_method (str): Method for generating channel swap probabilities.
                                         Options: 'normal', 'beta', 'uniform'.
            num_domain_classes (int): The total number of unique subjects in the entire dataset.
            enable_soft_labels (bool, optional): Whether to generate soft domain labels.
            sample_probability_method (str, optional): Method for generating sample swap probabilities.
                                                     Options: 'normal', 'beta', 'uniform'.
            use_legacy_method (bool, optional): Whether to use the original balanced swapping method.
            debug (bool, optional): Enable debug mode for visualization.
        """
        self.channel_num = channel_num
        self.swap_probability_method = swap_probability_method
        self.sample_probability_method = sample_probability_method
        self.num_domain_classes = num_domain_classes
        self.enable_soft_labels = enable_soft_labels
        self.use_legacy_method = use_legacy_method
        self.debug = debug

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply subject-level channel swapping to input data.
        
        Args:
            data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Input tuple containing:
                - data_tensor: EEG data of shape (batch_size, channels, height, width)
                - y_labels: Classification labels
                - domain_labels: Subject/domain labels
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                If soft labels enabled: (augmented_data, soft_labels)
                Otherwise: just augmented_data
        """
        print(f"🔆 Using {'legacy' if self.use_legacy_method else 'probabilistic pool-based'} method for SubjectLevelChannelSwap")
        if self.use_legacy_method:
            return self._legacy_balanced_swap(data)
        else:
            return self._probabilistic_pool_based_swap(data)
    
    def _legacy_balanced_swap(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Original balanced swapping method preserved for compatibility.
        
        This method requires equal sample counts per subject and uses strict sequential matching.
        """
        data_tensor, y_labels, domain_labels = data
        
        if domain_labels is None:
            raise ValueError("domain_labels must be provided for SubjectLevelChannelSwap.")
        if y_labels is None:
            raise ValueError("y_labels must be provided for SubjectLevelChannelSwap.")
        
        augmented_data = data_tensor.clone()
        
        if self.enable_soft_labels:
            channel_source_counts = torch.zeros(data_tensor.shape[0], self.num_domain_classes, dtype=torch.long)
            channel_source_counts.scatter_(1, domain_labels.unsqueeze(1), self.channel_num)
        
        subjects_in_batch = torch.unique(domain_labels).tolist()
        if len(subjects_in_batch) < 2:
            if self.enable_soft_labels:
                return augmented_data, SoftLabelGenerator.from_channel_counts(channel_source_counts, self.channel_num)
            return augmented_data

        subject_indices = {sid: (domain_labels == sid).nonzero(as_tuple=True)[0] for sid in subjects_in_batch}
        channels_to_swap = _decide_swap(_get_swap_probabilities(self.channel_num, self.swap_probability_method))

        for channel_idx in torch.where(channels_to_swap)[0]:
            shuffled_subjects = random.sample(subjects_in_batch, len(subjects_in_batch))
            subject_map = {orig: shuffled for orig, shuffled in zip(subjects_in_batch, shuffled_subjects)}
            temp_channel_data = augmented_data[:, channel_idx, :, :].clone()

            for orig_subject, target_subject in subject_map.items():
                orig_indices, target_indices = subject_indices[orig_subject], subject_indices[target_subject]
                if len(orig_indices) == len(target_indices):
                    for orig_idx, target_idx in zip(orig_indices, target_indices):
                        if y_labels[orig_idx] == y_labels[target_idx]:
                            augmented_data[target_idx, channel_idx, :, :] = temp_channel_data[orig_idx]
                            if self.enable_soft_labels:
                                channel_source_counts[target_idx, target_subject] -= 1
                                channel_source_counts[target_idx, orig_subject] += 1
                                
        if self.enable_soft_labels:
            return augmented_data, SoftLabelGenerator.from_channel_counts(channel_source_counts, self.channel_num)

        return augmented_data
    
    def _probabilistic_pool_based_swap(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        New probabilistic pool-based swapping method with dual probability gates.
        
        This method uses:
        1. Sample-level probability gates to determine which samples participate in swapping
        2. Channel-level probability gates to determine which channels to swap
        3. Pool-based subject grouping for efficient matching
        4. Flexible swapping without strict sample count requirements
        """
        data_tensor, y_labels, domain_labels = data
        
        if domain_labels is None:
            raise ValueError("domain_labels must be provided for SubjectLevelChannelSwap.")
        if y_labels is None:
            raise ValueError("y_labels must be provided for SubjectLevelChannelSwap.")
        
        augmented_data = data_tensor.clone()
        
        # Initialize soft label tracking
        if self.enable_soft_labels:
            channel_source_counts = torch.zeros(data_tensor.shape[0], self.num_domain_classes, dtype=torch.long)
            channel_source_counts.scatter_(1, domain_labels.unsqueeze(1), self.channel_num)
        
        subjects_in_batch = torch.unique(domain_labels).tolist()
        if len(subjects_in_batch) < 2:
            if self.enable_soft_labels:
                return augmented_data, SoftLabelGenerator.from_channel_counts(channel_source_counts, self.channel_num)
            return augmented_data

        # === STEP 1: Generate sample-level swap probabilities ===
        sample_swap_probabilities = _get_swap_probabilities(
            num_channels=data_tensor.shape[0],  # Using batch_size as num_samples
            method=self.sample_probability_method
        )
        sample_swap_decisions = _decide_swap(sample_swap_probabilities)
        
        # === STEP 2: Generate channel-level swap probabilities ===
        channel_swap_probabilities = _get_swap_probabilities(
            num_channels=self.channel_num,
            method=self.swap_probability_method
        )
        channel_swap_decisions = _decide_swap(channel_swap_probabilities)
        
        # === STEP 3: Build class-subject pools ===
        class_subject_pools = defaultdict(lambda: defaultdict(list))
        for idx, (cls, subj) in enumerate(zip(y_labels, domain_labels)):
            class_subject_pools[cls.item()][subj.item()].append(idx)
        
        # Store original channel data for swapping
        temp_channel_data = augmented_data.clone()
        
        # === STEP 4: Perform probabilistic swapping with dual gates ===
        for sample_idx in range(data_tensor.shape[0]):
            
            # 🎲 Sample-level Gate: Should this sample participate in swapping?
            if not sample_swap_decisions[sample_idx]:
                continue  # Skip this sample
                
            current_class = y_labels[sample_idx].item()
            current_subject = domain_labels[sample_idx].item()
            
            # Find other subjects with same class samples
            other_subject_pools = {
                subj: indices for subj, indices in class_subject_pools[current_class].items()
                if subj != current_subject and len(indices) > 0
            }
            
            if not other_subject_pools:
                continue  # No other subjects available for swapping
                
            for channel_idx in range(self.channel_num):
                
                # 🎲 Channel-level Gate: Should this channel be swapped?
                if not channel_swap_decisions[channel_idx]:
                    continue  # Skip this channel
                    
                # Randomly select source subject and sample
                source_subject = random.choice(list(other_subject_pools.keys()))
                source_sample_idx = random.choice(other_subject_pools[source_subject])
                
                # Perform the actual data swap
                augmented_data[sample_idx, channel_idx, :, :] = temp_channel_data[source_sample_idx, channel_idx, :, :]
                
                # Update soft label counts if enabled
                if self.enable_soft_labels:
                    channel_source_counts[sample_idx, current_subject] -= 1
                    channel_source_counts[sample_idx, source_subject] += 1
                                
        if self.enable_soft_labels:
            return augmented_data, SoftLabelGenerator.from_channel_counts(channel_source_counts, self.channel_num)

        return augmented_data

# --- Main Augmentation Wrapper ---

class ChannelwiseDataAugmentation(torch.nn.Module):
    """
    Unified Pipeline for EEG Channel-wise Data Augmentation.
    
    This class serves as a comprehensive wrapper for applying multiple channel-wise
    augmentation techniques to EEG data. It follows the PyTorch nn.Module pattern,
    making it easy to integrate into existing training pipelines and compose with
    other transformations.
    
    Key Features:
    - Sequential application of multiple augmentation methods
    - Automatic handling of data and domain label propagation
    - Type validation for supported augmentation methods
    - Seamless integration with PyTorch training loops
    
    Supported Augmentation Methods:
    - CorticalRegionChannelSwap: Anatomically-informed channel swapping
    - SubjectLevelChannelSwap: Cross-subject channel data mixing
    
    Design Philosophy:
    The wrapper maintains a clean separation between different augmentation strategies
    while providing a unified interface. Each augmentation method receives the same
    input format (data, domain_labels) and can decide whether to use domain information.
    
    Example Usage:
        >>> # Single augmentation
        >>> cortical_aug = CorticalRegionChannelSwap(64, regions, 'uniform')
        >>> pipeline = ChannelwiseDataAugmentation(cortical_aug)
        >>> 
        >>> # Multiple augmentations
        >>> subject_aug = SubjectLevelChannelSwap(64, 'beta')
        >>> pipeline = ChannelwiseDataAugmentation([cortical_aug, subject_aug])
        >>> 
        >>> # Apply in training loop
        >>> augmented_data = pipeline(data, domain_labels)
    """
    
    def __init__(self, augmentation_methods: Union[object, List[object]]):
        """
        Initialize the augmentation pipeline.
        
        Args:
            augmentation_methods: Either a single augmentation instance or a list of them.
                                 All methods must be instances of supported augmentation classes.
                                 The order in the list determines the application sequence.
        
        Raises:
            ValueError: If any augmentation method is not of a supported type.
            TypeError: If augmentation_methods is neither a supported type nor a list.
        
        Note:
            The pipeline applies augmentations sequentially, so the order matters.
            Earlier augmentations may affect the input to later ones.
        """
        super(ChannelwiseDataAugmentation, self).__init__()
        
        # Normalize input to list format
        if not isinstance(augmentation_methods, list):
            augmentation_methods = [augmentation_methods]
            
        # Validate and store augmentation methods
        self.augmentation_methods = []
        supported_types = (CorticalRegionChannelSwap, SubjectLevelChannelSwap)
        
        for i, method in enumerate(augmentation_methods):
            if isinstance(method, supported_types):
                self.augmentation_methods.append(method)
            else:
                raise ValueError(
                    f"Augmentation method at index {i} is of type {type(method).__name__}, "
                    f"but only {[cls.__name__ for cls in supported_types]} are supported."
                )

    def forward(self, data: torch.Tensor, y_labels: torch.Tensor = None, domain_labels: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply the configured augmentation pipeline to input data.
        
        Args:
            data (torch.Tensor): Input EEG data tensor
            y_labels (torch.Tensor, optional): Classification labels
            domain_labels (torch.Tensor, optional): Subject/domain labels
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                If any method returns soft labels: (augmented_data, soft_labels)
                Otherwise: just augmented_data
        """
        augmented_data = (data, y_labels, domain_labels)
        soft_labels = None
        
        for method in self.augmentation_methods:
            result = method(augmented_data)
            
            # Handle methods that return soft labels
            if isinstance(result, tuple) and len(result) == 2:
                augmented_data = (result[0], y_labels, domain_labels)
                soft_labels = result[1]
            else:
                augmented_data = (result, y_labels, domain_labels)
            
        final_data = augmented_data[0]
        return (final_data, soft_labels) if soft_labels is not None else final_data