"""
Cross-dataset t-SNE visualization for raw3 vs raw5&6 data comparison.

This module provides functions to compare the distribution of data between
raw3 (wire) and raw5&6 (wireless) datasets using t-SNE visualization.
"""

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as path
from scipy import signal
import seaborn as sns
import itertools
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_psd(data: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Computes the Power Spectral Density (PSD) of the data.

    Args:
        data (np.array): The input data with shape [samples, channels, time_points]
        fs (int): Sampling frequency.

    Returns:
        np.array: The PSD of the data.
    """
    f, Pxx = signal.welch(data, fs=fs)
    return Pxx


def load_dataset_data(data_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load all .npy files from a dataset directory and organize them by subject.
    
    Args:
        data_path (str): Path to the dataset directory
        
    Returns:
        Dict[str, List[np.ndarray]]: Dictionary with subject names as keys and 
                                   [day1_data, day8_data] as values
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Dataset path does not exist: {data_path}")
    
    npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_path}")
    
    data_dict = {}
    
    for npy_file in npy_files:
        # Parse filename: subject_condition_day.npy
        parts = path.splitext(npy_file)[0].split('_')
        subject_name = parts[0]
        condition = parts[1]  # 'before' or 'after'
        day = parts[2]  # 'day1' or 'day8'
        
        # Load data
        data = np.load(path.join(data_path, npy_file))
        
        # Initialize subject entry if not exists
        if subject_name not in data_dict:
            data_dict[subject_name] = [None, None]  # [day1/before, day8/after]
        
        # Store data based on condition
        if condition == 'before' or day == 'day1':
            data_dict[subject_name][0] = data
        elif condition == 'after' or day == 'day8':
            data_dict[subject_name][1] = data
    
    # Remove subjects with incomplete data
    complete_subjects = {}
    for subject, data_list in data_dict.items():
        if data_list[0] is not None and data_list[1] is not None:
            complete_subjects[subject] = data_list
        else:
            print(f"Warning: Subject {subject} has incomplete data, skipping...")
    
    print(f"Loaded {len(complete_subjects)} complete subjects from {data_path}")
    for subject, data_list in complete_subjects.items():
        print(f"  {subject}: Day1 {data_list[0].shape}, Day8 {data_list[1].shape}")
    
    return complete_subjects


def prepare_psd_data(data_dict: Dict[str, List[np.ndarray]], dataset_name: str) -> Dict[str, List[np.ndarray]]:
    """
    Convert EEG data to Power Spectral Density (PSD) format.
    
    Args:
        data_dict: Dictionary with subject data
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        Dictionary with PSD data in the same format
    """
    psd_data_dict = {}
    
    print(f"\nComputing PSD for {dataset_name} dataset...")
    
    for subject, data_list in data_dict.items():
        psd_data_dict[subject] = []
        
        for i, data in enumerate(data_list):
            day_name = "Day1" if i == 0 else "Day8"
            psd = compute_psd(data)
            # Reshape to [samples, features]
            psd_reshaped = psd.reshape(psd.shape[0], -1)
            psd_data_dict[subject].append(psd_reshaped)
            
            print(f"  {subject} {day_name}: {data.shape} -> PSD {psd_reshaped.shape}")
    
    return psd_data_dict


def create_combined_dataset(raw3_psd: Dict, raw5_psd: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Combine data from both datasets for cross-dataset comparison.
    
    Args:
        raw3_psd: PSD data from raw3 dataset
        raw5_psd: PSD data from raw5&6 dataset
        
    Returns:
        Tuple of (combined_data, subject_labels, dataset_labels, subject_mapping)
    """
    all_data = []
    subject_labels = []
    dataset_labels = []
    day_labels = []
    
    # Create unified subject mapping
    all_subjects = list(raw3_psd.keys()) + list(raw5_psd.keys())
    subject_to_id = {subject: idx for idx, subject in enumerate(all_subjects)}
    
    # Add raw3 data (dataset_id = 0)
    print("\nAdding raw3 (wire) data...")
    for subject, psd_data_list in raw3_psd.items():
        subject_id = subject_to_id[subject]
        
        for day_idx, psd_data in enumerate(psd_data_list):
            all_data.append(psd_data)
            subject_labels.extend([subject_id] * psd_data.shape[0])
            dataset_labels.extend([0] * psd_data.shape[0])  # 0 for raw3
            day_labels.extend([day_idx] * psd_data.shape[0])  # 0 for day1, 1 for day8
            
            print(f"  {subject} Day{day_idx+1}: {psd_data.shape[0]} samples")
    
    # Add raw5&6 data (dataset_id = 1)
    print("\nAdding raw5&6 (wireless) data...")
    for subject, psd_data_list in raw5_psd.items():
        subject_id = subject_to_id[subject]
        
        for day_idx, psd_data in enumerate(psd_data_list):
            all_data.append(psd_data)
            subject_labels.extend([subject_id] * psd_data.shape[0])
            dataset_labels.extend([1] * psd_data.shape[0])  # 1 for raw5&6
            day_labels.extend([day_idx] * psd_data.shape[0])
            
            print(f"  {subject} Day{day_idx+1}: {psd_data.shape[0]} samples")
    
    # Concatenate all data
    combined_data = np.concatenate(all_data, axis=0)
    subject_labels = np.array(subject_labels)
    dataset_labels = np.array(dataset_labels)
    day_labels = np.array(day_labels)
    
    print(f"\nCombined dataset shape: {combined_data.shape}")
    print(f"Subject labels shape: {subject_labels.shape}, unique subjects: {len(np.unique(subject_labels))}")
    print(f"Dataset labels shape: {dataset_labels.shape}, raw3: {np.sum(dataset_labels==0)}, raw5&6: {np.sum(dataset_labels==1)}")
    print(f"Day labels shape: {day_labels.shape}, day1: {np.sum(day_labels==0)}, day8: {np.sum(day_labels==1)}")
    
    return combined_data, subject_labels, dataset_labels, day_labels, subject_to_id


def compute_cross_dataset_tsne(
    combined_data: np.ndarray
) -> np.ndarray:
    """
    Compute t-SNE transformation for cross-dataset analysis.
    
    Args:
        combined_data: Combined PSD data
        
    Returns:
        np.ndarray: t-SNE transformed data with shape (n_samples, 2)
    """
    print("\nRunning t-SNE on combined dataset...")
    print(f"Data shape: {combined_data.shape}")
    
    # Run t-SNE
    perplexity_value = min(30, combined_data.shape[0] - 1)
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity_value,
        # n_iter=1000,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(combined_data)
    
    print(f"t-SNE completed. Output shape: {tsne_results.shape}")
    return tsne_results


def plot_cross_dataset_tsne_comprehensive(
    tsne_results: np.ndarray,
    subject_labels: np.ndarray, 
    dataset_labels: np.ndarray, 
    day_labels: np.ndarray,
    subject_mapping: Dict,
    save_path: Optional[str] = None
):
    """
    Create comprehensive t-SNE visualization comparing raw3 vs raw5&6 datasets.
    
    Args:
        tsne_results: Pre-computed t-SNE results with shape (n_samples, 2)
        subject_labels: Subject ID labels
        dataset_labels: Dataset ID labels (0=raw3, 1=raw5&6)
        day_labels: Day labels (0=day1, 1=day8)
        subject_mapping: Subject name to ID mapping
        save_path: Optional path to save the plot
    """
    # Create reverse mapping for subject names
    id_to_subject = {v: k for k, v in subject_mapping.items()}
    
    # Set up the plot style
    sns.set_theme(style="white", context="notebook")
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(24, 12))
    
    # Define colors and markers
    dataset_colors = {0: '#FF6B6B', 1: '#4ECDC4'}  # raw3: red, raw5&6: teal
    dataset_names = {0: 'raw3 (wire)', 1: 'raw5&6 (wireless)'}
    day_colors = {0: '#9CBBE9', 1: '#E6B89D'}  # day1: blue, day8: orange  
    day_names = {0: 'Day1 (before)', 1: 'Day8 (after)'}
    
    # Fixed colors for subjects
    raw3_colors = ["#A2E1DB", "#E6B89D", '#869ECB']  # Red variations
    raw56_colors = ['#A2E1DB', '#E6B89D', '#869ECB']  # Teal/green variations
    
    # Get unique subjects
    unique_subjects = np.unique(subject_labels)
    
    # Plot 1: Dataset comparison (top-left)
    ax1 = axes[0, 0]
    for dataset_id in [0, 1]:
        mask = dataset_labels == dataset_id
        ax1.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=dataset_colors[dataset_id], 
            label=dataset_names[dataset_id],
            alpha=0.7, s=60, edgecolors='black', linewidth=0.5
        )
    ax1.set_title('Dataset Comparison: raw3 vs raw5&6', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(ax=ax1, left=True, bottom=True)
    
    # Plot 2: Day comparison (top-right)
    ax2 = axes[0, 1]
    for day_id in [0, 1]:
        mask = day_labels == day_id
        ax2.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=day_colors[day_id],
            label=day_names[day_id],
            alpha=0.7, s=60, edgecolors='black', linewidth=0.5
        )
    ax2.set_title('Day Comparison: Day1 vs Day8', fontsize=14, fontweight='bold')
    ax2.legend(frameon=False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, bottom=True)
    
    # Plot 3: Subject comparison within datasets (bottom-left)
    ax3 = axes[1, 0]
    # Plot raw3 subjects with different colors for each subject
    raw3_subjects = [sid for sid in unique_subjects if any(dataset_labels[subject_labels == sid] == 0)]
    for i, sid in enumerate(raw3_subjects):
        mask = subject_labels == sid
        subject_name = id_to_subject[sid]
        color_idx = i % len(raw3_colors)  # Cycle through colors if more subjects than colors
        ax3.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=raw3_colors[color_idx],
            label=f'{subject_name} (raw3)',
            alpha=0.7, s=80, edgecolors='black', linewidth=0.5
        )
    ax3.set_title('raw3 (wire) Subjects', fontsize=14, fontweight='bold')
    ax3.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xticks([])
    ax3.set_yticks([])
    sns.despine(ax=ax3, left=True, bottom=True)
    
    # Plot 4: Subject comparison within datasets (bottom-right)
    ax4 = axes[1, 1]
    # Plot raw5&6 subjects with different colors for each subject
    raw56_subjects = [sid for sid in unique_subjects if any(dataset_labels[subject_labels == sid] == 1)]
    for i, sid in enumerate(raw56_subjects):
        mask = subject_labels == sid
        subject_name = id_to_subject[sid]
        color_idx = i % len(raw56_colors)  # Cycle through colors if more subjects than colors
        ax4.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=raw56_colors[color_idx],
            label=f'{subject_name} (raw5&6)',
            alpha=0.7, s=80, edgecolors='black', linewidth=0.5
        )
    ax4.set_title('raw5&6 (wireless) Subjects', fontsize=14, fontweight='bold')
    ax4.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_xticks([])
    ax4.set_yticks([])
    sns.despine(ax=ax4, left=True, bottom=True)
    
    
    # Plot 5: Dataset comparison with class distinction (bottom)
    ax5 = axes[2, 0]
    # Define markers for different days/classes
    day_markers = {0: 'o', 1: 's'}  # circle for day1, square for day8
    
    for dataset_id in [0, 1]:
        for day_id in [0, 1]:
            mask = (dataset_labels == dataset_id) & (day_labels == day_id)
            if np.sum(mask) > 0:
                ax5.scatter(
                    tsne_results[mask, 0], tsne_results[mask, 1],
                    c=dataset_colors[dataset_id], 
                    marker=day_markers[day_id],
                    alpha=0.7, s=80, edgecolors='black', linewidth=0.5,
                    label=f'{dataset_names[dataset_id]} - {day_names[day_id]}'
                )
    
    ax5.set_title('Dataset and Day Comparison', fontsize=14, fontweight='bold')
    ax5.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticks([])
    ax5.set_yticks([])
    sns.despine(ax=ax5, left=True, bottom=True)
    
    # Hide the bottom-right subplot since we only need 5 plots
    axes[2, 1].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        # Save comprehensive plot
        plt.savefig(f"{save_path}/cross_dataset_tsne_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_tsne_comprehensive.svg", format='svg', bbox_inches='tight')
        print(f"Comprehensive plot saved to {save_path}")
        
        # Save individual plots
        print("Saving individual plots...")
        
        # Save plot 1: Dataset comparison
        fig1, ax1_single = plt.subplots(1, 1, figsize=(8, 6))
        for dataset_id in [0, 1]:
            mask = dataset_labels == dataset_id
            ax1_single.scatter(
                tsne_results[mask, 0], tsne_results[mask, 1],
                c=dataset_colors[dataset_id], 
                label=dataset_names[dataset_id],
                alpha=0.7, s=60, edgecolors='black', linewidth=0.5
            )
        ax1_single.set_title('Dataset Comparison: raw3 vs raw5&6', fontsize=14, fontweight='bold')
        ax1_single.legend(frameon=False)
        ax1_single.set_xticks([])
        ax1_single.set_yticks([])
        sns.despine(ax=ax1_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cross_dataset_1_dataset_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_1_dataset_comparison.svg", format='svg', bbox_inches='tight')
        plt.close(fig1)
        
        # Save plot 2: Day comparison
        fig2, ax2_single = plt.subplots(1, 1, figsize=(8, 6))
        for day_id in [0, 1]:
            mask = day_labels == day_id
            ax2_single.scatter(
                tsne_results[mask, 0], tsne_results[mask, 1],
                c=day_colors[day_id],
                label=day_names[day_id],
                alpha=0.7, s=60, edgecolors='black', linewidth=0.5
            )
        ax2_single.set_title('Day Comparison: Day1 vs Day8', fontsize=14, fontweight='bold')
        ax2_single.legend(frameon=False)
        ax2_single.set_xticks([])
        ax2_single.set_yticks([])
        sns.despine(ax=ax2_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cross_dataset_2_day_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_2_day_comparison.svg", format='svg', bbox_inches='tight')
        plt.close(fig2)
        
        # Save plot 3: raw3 subjects
        fig3, ax3_single = plt.subplots(1, 1, figsize=(10, 6))
        raw3_subjects = [sid for sid in unique_subjects if any(dataset_labels[subject_labels == sid] == 0)]
        for i, sid in enumerate(raw3_subjects):
            mask = subject_labels == sid
            subject_name = id_to_subject[sid]
            color_idx = i % len(raw3_colors)
            ax3_single.scatter(
                tsne_results[mask, 0], tsne_results[mask, 1],
                c=raw3_colors[color_idx],
                label=f'{subject_name} (raw3)',
                alpha=0.7, s=80, edgecolors='black', linewidth=0.5
            )
        ax3_single.set_title('raw3 (wire) Subjects', fontsize=14, fontweight='bold')
        ax3_single.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3_single.set_xticks([])
        ax3_single.set_yticks([])
        sns.despine(ax=ax3_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cross_dataset_3_raw3_subjects.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_3_raw3_subjects.svg", format='svg', bbox_inches='tight')
        plt.close(fig3)
        
        # Save plot 4: raw5&6 subjects
        fig4, ax4_single = plt.subplots(1, 1, figsize=(10, 6))
        raw56_subjects = [sid for sid in unique_subjects if any(dataset_labels[subject_labels == sid] == 1)]
        for i, sid in enumerate(raw56_subjects):
            mask = subject_labels == sid
            subject_name = id_to_subject[sid]
            color_idx = i % len(raw56_colors)
            ax4_single.scatter(
                tsne_results[mask, 0], tsne_results[mask, 1],
                c=raw56_colors[color_idx],
                label=f'{subject_name} (raw5&6)',
                alpha=0.7, s=80, edgecolors='black', linewidth=0.5
            )
        ax4_single.set_title('raw5&6 (wireless) Subjects', fontsize=14, fontweight='bold')
        ax4_single.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4_single.set_xticks([])
        ax4_single.set_yticks([])
        sns.despine(ax=ax4_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cross_dataset_4_raw56_subjects.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_4_raw56_subjects.svg", format='svg', bbox_inches='tight')
        plt.close(fig4)
        
        # Save plot 5: Dataset and day comparison
        fig5, ax5_single = plt.subplots(1, 1, figsize=(10, 6))
        day_markers = {0: 'o', 1: 's'}
        for dataset_id in [0, 1]:
            for day_id in [0, 1]:
                mask = (dataset_labels == dataset_id) & (day_labels == day_id)
                if np.sum(mask) > 0:
                    ax5_single.scatter(
                        tsne_results[mask, 0], tsne_results[mask, 1],
                        c=dataset_colors[dataset_id], 
                        marker=day_markers[day_id],
                        alpha=0.7, s=80, edgecolors='black', linewidth=0.5,
                        label=f'{dataset_names[dataset_id]} - {day_names[day_id]}'
                    )
        ax5_single.set_title('Dataset and Day Comparison', fontsize=14, fontweight='bold')
        ax5_single.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5_single.set_xticks([])
        ax5_single.set_yticks([])
        sns.despine(ax=ax5_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/cross_dataset_5_dataset_day_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_dataset_5_dataset_day_comparison.svg", format='svg', bbox_inches='tight')
        plt.close(fig5)
        
        print(f"Saved 5 individual cross-dataset plots to {save_path}")
    
    plt.show()


def plot_cross_subject_combinations(
    raw3_psd: Dict, 
    raw5_psd: Dict,
    save_path: Optional[str] = None
):
    """
    Create t-SNE plots for all possible cross-dataset subject combinations.
    
    Args:
        raw3_psd: PSD data from raw3 dataset
        raw5_psd: PSD data from raw5&6 dataset
        save_path: Optional path to save the plots
    """
    raw3_subjects = list(raw3_psd.keys())
    raw56_subjects = list(raw5_psd.keys())
    
    # Generate all combinations of raw3 vs raw5&6 subjects
    combinations = list(itertools.product(raw3_subjects, raw56_subjects))
    
    print(f"\nGenerating t-SNE plots for {len(combinations)} cross-dataset subject combinations...")
    
    # Calculate grid dimensions
    n_combinations = len(combinations)
    n_cols = 3  # 3 columns for better layout
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_combinations == 1:
        axes = axes.reshape(1, 1)
    
    # Set plot style
    sns.set_theme(style="white", context="notebook")
    
    # Define colors and markers
    dataset_colors = {'raw3': '#FF6B6B', 'raw5&6': '#4ECDC4'}
    day_colors = {'Day1': '#9CBBE9', 'Day8': '#E6B89D'}
    
    for idx, (raw3_subj, raw56_subj) in enumerate(combinations):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        print(f"Processing combination {idx+1}/{len(combinations)}: {raw3_subj} vs {raw56_subj}")
        
        # Prepare data for this combination
        data_to_combine = []
        dataset_labels = []
        day_labels = []
        
        # Add raw3 subject data
        for day_idx, day_data in enumerate(raw3_psd[raw3_subj]):
            data_to_combine.append(day_data)
            dataset_labels.extend(['raw3'] * day_data.shape[0])
            day_labels.extend([f'Day{day_idx+1}'] * day_data.shape[0])
        
        # Add raw5&6 subject data
        for day_idx, day_data in enumerate(raw5_psd[raw56_subj]):
            data_to_combine.append(day_data)
            dataset_labels.extend(['raw5&6'] * day_data.shape[0])
            day_labels.extend([f'Day{day_idx+1}'] * day_data.shape[0])
        
        # Combine data
        combined_data = np.concatenate(data_to_combine, axis=0)
        
        # Run t-SNE for this combination
        perplexity_value = min(20, combined_data.shape[0] - 1)
        if perplexity_value < 2:
            ax.text(0.5, 0.5, f"Not enough\nsamples", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{raw3_subj} vs {raw56_subj}")
            continue
        
        tsne_combo = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity_value,
            # n_iter=500,  # Reduced iterations for faster processing
            init='pca',
            learning_rate='auto'
        )
        tsne_results = tsne_combo.fit_transform(combined_data)
        
        # Create scatter plot with both dataset and day information
        for i, (x, y) in enumerate(tsne_results):
            dataset = dataset_labels[i]
            day = day_labels[i]
            
            # Choose marker based on day
            marker = 'o' if day == 'Day1' else 's'
            
            ax.scatter(x, y, 
                      c=dataset_colors[dataset], 
                      marker=marker,
                      alpha=0.7, s=60, 
                      edgecolors='black', linewidth=0.3)
        
        ax.set_title(f"{raw3_subj} vs {raw56_subj}", fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    
    # Hide unused subplots
    for idx in range(n_combinations, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Create legend
    legend_elements = []
    # Dataset colors
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dataset_colors['raw3'], 
               markersize=10, label='raw3 (wire)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dataset_colors['raw5&6'], 
               markersize=10, label='raw5&6 (wireless)')
    ])
    # Day markers
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Day1 (before)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Day8 (after)')
    ])
    
    fig.legend(handles=legend_elements, 
              bbox_to_anchor=(0.98, 0.98), 
              loc='upper right',
              frameon=False, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(f"{save_path}/cross_subject_combinations_tsne.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/cross_subject_combinations_tsne.svg", format='svg', bbox_inches='tight')
        print(f"Cross-subject combinations plot saved to {save_path}")
    
    plt.show()


def calculate_silhouette_scores(
    tsne_results: np.ndarray,
    dataset_labels: np.ndarray,
    subject_labels: np.ndarray = None,
    day_labels: np.ndarray = None,
    subject_mapping: dict = None,
    save_path: str = None
) -> dict:
    """
    Calculate silhouette scores for t-SNE results to evaluate separation quality
    between wireless (raw5&6) and wired (raw3) datasets.
    
    Args:
        tsne_results: t-SNE transformed 2D coordinates with shape (n_samples, 2)
        dataset_labels: Dataset labels (0=raw3/wire, 1=raw5&6/wireless)
        subject_labels: Subject ID labels (optional)
        day_labels: Day labels (0=day1, 1=day8) (optional)
        subject_mapping: Subject name to ID mapping (optional)
        save_path: Path to save results (optional)
        
    Returns:
        Dictionary containing various silhouette score analyses
    """
    print("=== Silhouette Score Analysis ===")
    
    results = {}
    
    # 1. Overall silhouette score (wireless vs wired)
    print("\n1. Overall Dataset Separation Quality")
    overall_score = silhouette_score(tsne_results, dataset_labels)
    results['overall_score'] = overall_score
    print(f"   Overall Silhouette Score: {overall_score:.4f}")
    print(f"   Interpretation: {'Excellent' if overall_score > 0.7 else 'Good' if overall_score > 0.5 else 'Fair' if overall_score > 0.25 else 'Poor'} separation")
    
    # 2. Individual sample silhouette scores
    sample_scores = silhouette_samples(tsne_results, dataset_labels)
    results['sample_scores'] = sample_scores
    
    # Calculate statistics by dataset
    raw3_mask = dataset_labels == 0
    raw56_mask = dataset_labels == 1
    
    raw3_scores = sample_scores[raw3_mask]
    raw56_scores = sample_scores[raw56_mask]
    
    print(f"\n   raw3 (wire) samples:")
    print(f"     Mean silhouette: {np.mean(raw3_scores):.4f} ± {np.std(raw3_scores):.4f}")
    print(f"     Min/Max: {np.min(raw3_scores):.4f} / {np.max(raw3_scores):.4f}")
    
    print(f"\n   raw5&6 (wireless) samples:")
    print(f"     Mean silhouette: {np.mean(raw56_scores):.4f} ± {np.std(raw56_scores):.4f}")
    print(f"     Min/Max: {np.min(raw56_scores):.4f} / {np.max(raw56_scores):.4f}")
    
    results['raw3_stats'] = {
        'mean': np.mean(raw3_scores),
        'std': np.std(raw3_scores),
        'min': np.min(raw3_scores),
        'max': np.max(raw3_scores)
    }
    
    results['raw56_stats'] = {
        'mean': np.mean(raw56_scores),
        'std': np.std(raw56_scores),
        'min': np.min(raw56_scores),
        'max': np.max(raw56_scores)
    }
    
    # 3. Subject-wise analysis (if subject labels provided)
    if subject_labels is not None and subject_mapping is not None:
        print("\n2. Subject-wise Separation Quality")
        subject_scores = {}
        subject_sample_scores = {}
        id_to_subject = {v: k for k, v in subject_mapping.items()}
        
        for subject_id in np.unique(subject_labels):
            subject_mask = subject_labels == subject_id
            subject_name = id_to_subject.get(subject_id, f"Subject_{subject_id}")
            
            if np.sum(subject_mask) > 1:  # Need at least 2 samples
                subject_tsne = tsne_results[subject_mask]
                subject_datasets = dataset_labels[subject_mask]
                subject_sample_silhouettes = sample_scores[subject_mask]
                
                # Store individual sample scores for this subject
                subject_sample_scores[subject_name] = {
                    'sample_scores': subject_sample_silhouettes,
                    'dataset_labels': subject_datasets,
                    'mean_score': np.mean(subject_sample_silhouettes),
                    'std_score': np.std(subject_sample_silhouettes),
                    'min_score': np.min(subject_sample_silhouettes),
                    'max_score': np.max(subject_sample_silhouettes),
                    'total_samples': len(subject_sample_silhouettes)
                }
                
                # Only calculate if subject has both dataset types
                if len(np.unique(subject_datasets)) > 1:
                    subj_score = silhouette_score(subject_tsne, subject_datasets)
                    subject_scores[subject_name] = subj_score
                    print(f"   {subject_name}: {subj_score:.4f} (samples: {len(subject_sample_silhouettes)})")
                else:
                    # If subject only has one dataset type, use mean of sample scores
                    subject_scores[subject_name] = np.mean(subject_sample_silhouettes)
                    print(f"   {subject_name}: {np.mean(subject_sample_silhouettes):.4f}* (single dataset, samples: {len(subject_sample_silhouettes)})")
        
        results['subject_scores'] = subject_scores
        results['subject_sample_scores'] = subject_sample_scores
    
    # 4. Day-wise analysis (if day labels provided)
    if day_labels is not None:
        print("\n3. Day-wise Separation Quality")
        day_scores = {}
        
        for day_id in [0, 1]:
            day_mask = day_labels == day_id
            day_name = "Day1" if day_id == 0 else "Day8"
            
            if np.sum(day_mask) > 1:
                day_tsne = tsne_results[day_mask]
                day_datasets = dataset_labels[day_mask]
                
                # Only calculate if day has both dataset types
                if len(np.unique(day_datasets)) > 1:
                    day_score = silhouette_score(day_tsne, day_datasets)
                    day_scores[day_name] = day_score
                    print(f"   {day_name}: {day_score:.4f}")
        
        results['day_scores'] = day_scores
    
    # 5. Create results summary DataFrame
    summary_data = []
    summary_data.append(['Overall', overall_score, len(tsne_results), 'All data'])
    
    if 'subject_scores' in results:
        for subject, score in results['subject_scores'].items():
            subject_mask = subject_labels == subject_mapping[subject]
            sample_count = np.sum(subject_mask)
            summary_data.append(['Subject', score, sample_count, subject])
    
    if 'day_scores' in results:
        for day, score in results['day_scores'].items():
            day_id = 0 if day == "Day1" else 1
            day_mask = day_labels == day_id
            sample_count = np.sum(day_mask)
            summary_data.append(['Day', score, sample_count, day])
    
    summary_df = pd.DataFrame(summary_data, columns=['Analysis_Type', 'Silhouette_Score', 'Sample_Count', 'Category'])
    results['summary_df'] = summary_df
    
    print(f"\n4. Summary Table")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # 6. Save results if path provided
    if save_path:
        # Save detailed results to CSV
        summary_df.to_csv(f"{save_path}/silhouette_scores_summary.csv", index=False)
        print(f"\n   Results saved to {save_path}/silhouette_scores_summary.csv")
    
    return results


def plot_subject_silhouette_distribution(
    silhouette_results: dict,
    save_path: str = None
):
    """
    Create detailed subject-wise silhouette score distribution analysis.
    
    Args:
        silhouette_results: Results from calculate_silhouette_scores
        save_path: Path to save plots
    """
    if 'subject_sample_scores' not in silhouette_results:
        print("Subject sample scores not available in silhouette results")
        return
    
    subject_data = silhouette_results['subject_sample_scores']
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme
    dataset_colors = {0: '#FF6B6B', 1: '#4ECDC4'}  # raw3: red, raw5&6: teal
    dataset_names = {0: 'raw3 (wire)', 1: 'raw5&6 (wireless)'}
    
    # 1. Subject-wise silhouette score distribution (box plot) - top-left
    ax1 = axes[0, 0]
    
    subjects = list(subject_data.keys())
    subject_box_data = []
    subject_labels = []
    subject_colors = []
    
    for subject in subjects:
        scores = subject_data[subject]['sample_scores']
        datasets = subject_data[subject]['dataset_labels']
        
        # Separate by dataset for each subject
        raw3_scores = scores[datasets == 0]
        raw56_scores = scores[datasets == 1]
        
        if len(raw3_scores) > 0:
            subject_box_data.append(raw3_scores)
            subject_labels.append(f"{subject}\n(wire)")
            subject_colors.append(dataset_colors[0])
        
        if len(raw56_scores) > 0:
            subject_box_data.append(raw56_scores)
            subject_labels.append(f"{subject}\n(wireless)")
            subject_colors.append(dataset_colors[1])
    
    # Create box plot
    bp = ax1.boxplot(subject_box_data, labels=subject_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], subject_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Subject-wise Silhouette Score Distribution', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Silhouette Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Subject statistics summary (bar plot) - top-right
    ax2 = axes[0, 1]
    
    subject_means = [subject_data[s]['mean_score'] for s in subjects]
    subject_stds = [subject_data[s]['std_score'] for s in subjects]
    
    bars = ax2.bar(range(len(subjects)), subject_means, yerr=subject_stds, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    ax2.set_xticks(range(len(subjects)))
    ax2.set_xticklabels(subjects, rotation=45, ha='right')
    ax2.set_ylabel('Mean Silhouette Score')
    ax2.set_title('Subject Mean Silhouette Scores (±Std)', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, subject_means, subject_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Subject score range analysis - bottom-left
    ax3 = axes[1, 0]
    
    subject_ranges = [subject_data[s]['max_score'] - subject_data[s]['min_score'] for s in subjects]
    subject_mins = [subject_data[s]['min_score'] for s in subjects]
    subject_maxs = [subject_data[s]['max_score'] for s in subjects]
    
    # Plot range as error bars
    ax3.errorbar(range(len(subjects)), subject_means, 
                 yerr=[np.array(subject_means) - np.array(subject_mins),
                       np.array(subject_maxs) - np.array(subject_means)],
                 fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.8)
    
    ax3.set_xticks(range(len(subjects)))
    ax3.set_xticklabels(subjects, rotation=45, ha='right')
    ax3.set_ylabel('Silhouette Score Range')
    ax3.set_title('Subject Score Ranges (Min-Max)', fontweight='bold', fontsize=14)
    ax3.grid(alpha=0.3)
    
    # 4. Sample count vs score correlation - bottom-right
    ax4 = axes[1, 1]
    
    sample_counts = [subject_data[s]['total_samples'] for s in subjects]
    
    # Scatter plot of sample count vs mean score
    scatter = ax4.scatter(sample_counts, subject_means, 
                         s=100, alpha=0.7, c=subject_means, 
                         cmap='viridis', edgecolors='black')
    
    # Add subject labels
    for i, subject in enumerate(subjects):
        ax4.annotate(subject, (sample_counts[i], subject_means[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.8)
    
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Mean Silhouette Score')
    ax4.set_title('Sample Count vs Mean Score', fontweight='bold', fontsize=14)
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Mean Silhouette Score')
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(f"{save_path}/subject_silhouette_distribution.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/subject_silhouette_distribution.svg", format='svg', bbox_inches='tight')
        print(f"Subject silhouette distribution plot saved to {save_path}")
    
    plt.show()
    
    # Print detailed subject statistics
    print("\n=== Detailed Subject Statistics ===")
    for subject in subjects:
        data = subject_data[subject]
        print(f"\n{subject}:")
        print(f"  Total samples: {data['total_samples']}")
        print(f"  Mean ± Std: {data['mean_score']:.4f} ± {data['std_score']:.4f}")
        print(f"  Range: {data['min_score']:.4f} - {data['max_score']:.4f}")
        
        # Dataset breakdown
        datasets = data['dataset_labels']
        scores = data['sample_scores']
        raw3_count = np.sum(datasets == 0)
        raw56_count = np.sum(datasets == 1)
        
        if raw3_count > 0:
            raw3_scores = scores[datasets == 0]
            print(f"  raw3 (wire): {raw3_count} samples, mean {np.mean(raw3_scores):.4f}")
        
        if raw56_count > 0:
            raw56_scores = scores[datasets == 1]
            print(f"  raw5&6 (wireless): {raw56_count} samples, mean {np.mean(raw56_scores):.4f}")


def plot_silhouette_analysis(
    tsne_results: np.ndarray,
    dataset_labels: np.ndarray,
    silhouette_results: dict,
    save_path: str = None
):
    """
    Create comprehensive silhouette score visualization.
    
    Args:
        tsne_results: t-SNE transformed coordinates
        dataset_labels: Dataset labels (0=raw3, 1=raw5&6)
        silhouette_results: Results from calculate_silhouette_scores
        save_path: Path to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    # Color scheme
    dataset_colors = {0: '#FF6B6B', 1: '#4ECDC4'}  # raw3: red, raw5&6: teal
    dataset_names = {0: 'raw3 (wire)', 1: 'raw5&6 (wireless)'}
    sample_scores = silhouette_results['sample_scores']
    
    # 1a. raw3 t-SNE plot with silhouette scores as color intensity (top-left)
    ax1a = axes[0, 0]
    mask_raw3 = dataset_labels == 0
    scatter_raw3 = ax1a.scatter(
        tsne_results[mask_raw3, 0], tsne_results[mask_raw3, 1],
        c=sample_scores[mask_raw3], 
        cmap='viridis',
        alpha=0.7, s=60, 
        edgecolors='black', linewidth=0.3
    )
    ax1a.set_title('raw3 (wire) t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', fontweight='bold')
    ax1a.set_xticks([])
    ax1a.set_yticks([])
    plt.colorbar(scatter_raw3, ax=ax1a, label='Silhouette Score')
    
    # 1b. raw5&6 t-SNE plot with silhouette scores as color intensity (top-middle)
    ax1b = axes[0, 1]
    mask_raw56 = dataset_labels == 1
    scatter_raw56 = ax1b.scatter(
        tsne_results[mask_raw56, 0], tsne_results[mask_raw56, 1],
        c=sample_scores[mask_raw56], 
        cmap='viridis',
        alpha=0.7, s=60, 
        edgecolors='black', linewidth=0.3
    )
    ax1b.set_title('raw5&6 (wireless) t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', fontweight='bold')
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    plt.colorbar(scatter_raw56, ax=ax1b, label='Silhouette Score')
    
    # 2a. raw3 silhouette score distribution histogram (top-right)
    ax2a = axes[0, 2]
    raw3_scores = sample_scores[dataset_labels == 0]
    
    ax2a.hist(raw3_scores, bins=30, alpha=0.7, color='#FF6B6B', density=True)
    # ax2a.axvline(np.mean(raw3_scores), color='darkred', linestyle='--', 
    #              label=f'raw3 Mean: {np.mean(raw3_scores):.3f}')
    ax2a.set_xlabel('Silhouette Score')
    ax2a.set_ylabel('Density')
    ax2a.set_title('raw3 (wire) Silhouette Score Distribution', fontweight='bold')
    ax2a.legend()
    ax2a.grid(alpha=0.3)
    
    # 2b. raw5&6 silhouette score distribution histogram (bottom-left)
    ax2b = axes[1, 0]
    raw56_scores = sample_scores[dataset_labels == 1]
    
    ax2b.hist(raw56_scores, bins=30, alpha=0.7, color='#4ECDC4', density=True)
    # ax2b.axvline(np.mean(raw56_scores), color='darkgreen', linestyle='--', 
    #              label=f'raw5&6 Mean: {np.mean(raw56_scores):.3f}')
    ax2b.set_xlabel('Silhouette Score')
    ax2b.set_ylabel('Density')
    ax2b.set_title('raw5&6 (wireless) Silhouette Score Distribution', fontweight='bold')
    ax2b.legend()
    ax2b.grid(alpha=0.3)
    
    # 3. Subject-wise score distribution by dataset (bottom-middle)
    ax3 = axes[1, 1]
    if 'subject_sample_scores' in silhouette_results and silhouette_results['subject_sample_scores']:
        # Collect subject-wise scores by dataset
        raw3_subject_scores = []
        raw56_subject_scores = []
        
        for subject_name, subject_data in silhouette_results['subject_sample_scores'].items():
            dataset_labels_subj = subject_data['dataset_labels']
            sample_scores_subj = subject_data['sample_scores']
            
            # Get mean score for this subject in each dataset
            raw3_mask = dataset_labels_subj == 0
            raw56_mask = dataset_labels_subj == 1
            
            if np.sum(raw3_mask) > 0:
                raw3_subject_scores.append(np.mean(sample_scores_subj[raw3_mask]))
            if np.sum(raw56_mask) > 0:
                raw56_subject_scores.append(np.mean(sample_scores_subj[raw56_mask]))
        
        # Create histogram of subject-wise scores by dataset
        bins = np.linspace(
            min(min(raw3_subject_scores) if raw3_subject_scores else 0, 
                min(raw56_subject_scores) if raw56_subject_scores else 0) - 0.1,
            max(max(raw3_subject_scores) if raw3_subject_scores else 1, 
                max(raw56_subject_scores) if raw56_subject_scores else 1) + 0.1,
            15
        )
        
        if raw3_subject_scores:
            ax3.hist(raw3_subject_scores, bins=bins, alpha=0.7, color='#FF6B6B', 
                    label=f'raw3 (wire) subjects (n={len(raw3_subject_scores)})', density=True)
        if raw56_subject_scores:
            ax3.hist(raw56_subject_scores, bins=bins, alpha=0.7, color='#4ECDC4', 
                    label=f'raw5&6 (wireless) subjects (n={len(raw56_subject_scores)})', density=True)
        
        ax3.set_xlabel('Subject Mean Silhouette Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Subject-wise Scores by Dataset', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Subject-wise analysis\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Subject-wise Analysis', fontweight='bold')
    
    # 4. Summary statistics comparison (bottom-right)
    ax4 = axes[1, 2]
    categories = ['raw3 (wire)', 'raw5&6 (wireless)']
    means = [silhouette_results['raw3_stats']['mean'], silhouette_results['raw56_stats']['mean']]
    stds = [silhouette_results['raw3_stats']['std'], silhouette_results['raw56_stats']['std']]
    
    bars = ax4.bar(categories, means, yerr=stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Silhouette Score')
    ax4.set_title('Dataset Comparison\n(Mean ± Std)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        # Save comprehensive plot
        plt.savefig(f"{save_path}/silhouette_analysis_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_analysis_comprehensive.svg", format='svg', bbox_inches='tight')
        print(f"Silhouette analysis plot saved to {save_path}")
        
        # Save individual plots
        print("Saving individual plots...")
        
        # Save plot 1a: raw3 t-SNE with silhouette scores
        fig1a, ax1a_single = plt.subplots(1, 1, figsize=(8, 6))
        scatter_raw3_single = ax1a_single.scatter(
            tsne_results[mask_raw3, 0], tsne_results[mask_raw3, 1],
            c=sample_scores[mask_raw3], 
            cmap='viridis',
            alpha=0.7, s=60, 
            edgecolors='black', linewidth=0.3
        )
        ax1a_single.set_title('raw3 (wire) t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', fontweight='bold')
        ax1a_single.set_xticks([])
        ax1a_single.set_yticks([])
        plt.colorbar(scatter_raw3_single, ax=ax1a_single, label='Silhouette Score')
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_1a_raw3_tsne.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_1a_raw3_tsne.svg", format='svg', bbox_inches='tight')
        plt.close(fig1a)
        
        # Save plot 1b: raw5&6 t-SNE with silhouette scores
        fig1b, ax1b_single = plt.subplots(1, 1, figsize=(8, 6))
        scatter_raw56_single = ax1b_single.scatter(
            tsne_results[mask_raw56, 0], tsne_results[mask_raw56, 1],
            c=sample_scores[mask_raw56], 
            cmap='viridis',
            alpha=0.7, s=60, 
            edgecolors='black', linewidth=0.3
        )
        ax1b_single.set_title('raw5&6 (wireless) t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', fontweight='bold')
        ax1b_single.set_xticks([])
        ax1b_single.set_yticks([])
        plt.colorbar(scatter_raw56_single, ax=ax1b_single, label='Silhouette Score')
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_1b_raw56_tsne.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_1b_raw56_tsne.svg", format='svg', bbox_inches='tight')
        plt.close(fig1b)
        
        # Save plot 2a: raw3 histogram
        fig2a, ax2a_single = plt.subplots(1, 1, figsize=(8, 6))
        raw3_scores_single = sample_scores[dataset_labels == 0]
        ax2a_single.hist(raw3_scores_single, bins=30, alpha=0.7, color='#FF6B6B', density=True)
        # ax2a_single.axvline(np.mean(raw3_scores_single), color='darkred', linestyle='--', 
        #                    label=f'raw3 Mean: {np.mean(raw3_scores_single):.3f}')
        ax2a_single.set_xlabel('Silhouette Score')
        ax2a_single.set_ylabel('Density')
        ax2a_single.set_title('raw3 (wire) Silhouette Score Distribution', fontweight='bold')
        ax2a_single.legend()
        ax2a_single.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_2a_raw3_histogram.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_2a_raw3_histogram.svg", format='svg', bbox_inches='tight')
        plt.close(fig2a)
        
        # Save plot 2b: raw5&6 histogram
        fig2b, ax2b_single = plt.subplots(1, 1, figsize=(8, 6))
        raw56_scores_single = sample_scores[dataset_labels == 1]
        ax2b_single.hist(raw56_scores_single, bins=30, alpha=0.7, color='#4ECDC4', density=True)
        # ax2b_single.axvline(np.mean(raw56_scores_single), color='darkgreen', linestyle='--', 
        #                    label=f'raw5&6 Mean: {np.mean(raw56_scores_single):.3f}')
        ax2b_single.set_xlabel('Silhouette Score')
        ax2b_single.set_ylabel('Density')
        ax2b_single.set_title('raw5&6 (wireless) Silhouette Score Distribution', fontweight='bold')
        ax2b_single.legend()
        ax2b_single.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_2b_raw56_histogram.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_2b_raw56_histogram.svg", format='svg', bbox_inches='tight')
        plt.close(fig2b)
        
        # Save plot 3: Subject-wise analysis
        fig3, ax3_single = plt.subplots(1, 1, figsize=(8, 6))
        if 'subject_sample_scores' in silhouette_results and silhouette_results['subject_sample_scores']:
            # Collect subject-wise scores by dataset
            raw3_subject_scores = []
            raw56_subject_scores = []
            
            for subject_name, subject_data in silhouette_results['subject_sample_scores'].items():
                dataset_labels_subj = subject_data['dataset_labels']
                sample_scores_subj = subject_data['sample_scores']
                
                # Get mean score for this subject in each dataset
                raw3_mask = dataset_labels_subj == 0
                raw56_mask = dataset_labels_subj == 1
                
                if np.sum(raw3_mask) > 0:
                    raw3_subject_scores.append(np.mean(sample_scores_subj[raw3_mask]))
                if np.sum(raw56_mask) > 0:
                    raw56_subject_scores.append(np.mean(sample_scores_subj[raw56_mask]))
            
            # Create histogram of subject-wise scores by dataset
            bins = np.linspace(
                min(min(raw3_subject_scores) if raw3_subject_scores else 0, 
                    min(raw56_subject_scores) if raw56_subject_scores else 0) - 0.1,
                max(max(raw3_subject_scores) if raw3_subject_scores else 1, 
                    max(raw56_subject_scores) if raw56_subject_scores else 1) + 0.1,
                15
            )
            
            if raw3_subject_scores:
                ax3_single.hist(raw3_subject_scores, bins=bins, alpha=0.7, color='#FF6B6B', 
                               label=f'raw3 (wire) subjects (n={len(raw3_subject_scores)})', density=True)
            if raw56_subject_scores:
                ax3_single.hist(raw56_subject_scores, bins=bins, alpha=0.7, color='#4ECDC4', 
                               label=f'raw5&6 (wireless) subjects (n={len(raw56_subject_scores)})', density=True)
            
            ax3_single.set_xlabel('Subject Mean Silhouette Score')
            ax3_single.set_ylabel('Density')
            ax3_single.set_title('Distribution of Subject-wise Scores by Dataset', fontweight='bold')
            ax3_single.legend()
            ax3_single.grid(alpha=0.3)
        else:
            ax3_single.text(0.5, 0.5, 'Subject-wise analysis\nnot available', 
                           ha='center', va='center', transform=ax3_single.transAxes, fontsize=12)
            ax3_single.set_title('Subject-wise Analysis', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_3_subject_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_3_subject_analysis.svg", format='svg', bbox_inches='tight')
        plt.close(fig3)
        
        # Save plot 4: Summary statistics
        fig4, ax4_single = plt.subplots(1, 1, figsize=(8, 6))
        categories = ['raw3 (wire)', 'raw5&6 (wireless)']
        means = [silhouette_results['raw3_stats']['mean'], silhouette_results['raw56_stats']['mean']]
        stds = [silhouette_results['raw3_stats']['std'], silhouette_results['raw56_stats']['std']]
        
        bars = ax4_single.bar(categories, means, yerr=stds, capsize=5, 
                             color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
        ax4_single.set_ylabel('Mean Silhouette Score')
        ax4_single.set_title('Dataset Comparison\n(Mean ± Std)', fontweight='bold')
        ax4_single.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax4_single.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{save_path}/silhouette_4_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}/silhouette_4_summary.svg", format='svg', bbox_inches='tight')
        plt.close(fig4)
        
        print(f"Saved 6 individual silhouette analysis plots to {save_path}")
    
    plt.show()


def analyze_cross_dataset_distribution(
    raw3_path: str, 
    raw56_path: str, 
    save_path: Optional[str] = None,
    include_silhouette_analysis: bool = True
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Main function to analyze and visualize cross-dataset distribution differences.
    
    Args:
        raw3_path: Path to raw3 dataset directory
        raw56_path: Path to raw5&6 dataset directory  
        save_path: Optional path to save plots
        include_silhouette_analysis: Whether to include silhouette score analysis
        
    Returns:
        Tuple of (raw3_psd_data, raw56_psd_data, silhouette_results)
    """
    print("=== Cross-Dataset Distribution Analysis ===")
    print(f"raw3 path: {raw3_path}")
    print(f"raw5&6 path: {raw56_path}")
    
    # Load datasets
    print("\n1. Loading datasets...")
    raw3_data = load_dataset_data(raw3_path)
    raw56_data = load_dataset_data(raw56_path)
    
    # Convert to PSD
    print("\n2. Converting to PSD...")
    raw3_psd = prepare_psd_data(raw3_data, "raw3")
    raw56_psd = prepare_psd_data(raw56_data, "raw5&6")
    
    # Create comprehensive comparison
    print("\n3. Creating comprehensive dataset comparison...")
    combined_data, subject_labels, dataset_labels, day_labels, subject_mapping = create_combined_dataset(raw3_psd, raw56_psd)
    
    # Compute t-SNE
    tsne_results = compute_cross_dataset_tsne(combined_data)
    
    # Create visualization
    plot_cross_dataset_tsne_comprehensive(
        tsne_results, subject_labels, dataset_labels, day_labels, 
        subject_mapping, save_path
    )
    
    # Silhouette score analysis
    silhouette_results = None
    if include_silhouette_analysis:
        print("\n5. Calculating silhouette scores...")
        silhouette_results = calculate_silhouette_scores(
            tsne_results=tsne_results,
            dataset_labels=dataset_labels,
            subject_labels=subject_labels,
            day_labels=day_labels,
            subject_mapping=subject_mapping,
            save_path=save_path
        )
        
        print("\n6. Creating silhouette score visualization...")
        plot_silhouette_analysis(
            tsne_results=tsne_results,
            dataset_labels=dataset_labels,
            silhouette_results=silhouette_results,
            save_path=save_path
        )
        
        # Create detailed subject-wise silhouette distribution analysis
        print("\n7. Creating subject-wise silhouette distribution analysis...")
        plot_subject_silhouette_distribution(
            silhouette_results=silhouette_results,
            save_path=save_path
        )
        
        # Display key results summary
        print(f"\n🎯 Key Silhouette Analysis Results:")
        print(f"   Overall Silhouette Score: {silhouette_results['overall_score']:.4f}")
        print(f"   Total Samples: {len(tsne_results)}")
        print(f"   raw3 (wire) samples: {np.sum(dataset_labels == 0)}")
        print(f"   raw5&6 (wireless) samples: {np.sum(dataset_labels == 1)}")
        
        if 'subject_scores' in silhouette_results and silhouette_results['subject_scores']:
            best_subject = max(silhouette_results['subject_scores'].items(), key=lambda x: x[1])
            worst_subject = min(silhouette_results['subject_scores'].items(), key=lambda x: x[1])
            print(f"   Best subject separation: {best_subject[0]} ({best_subject[1]:.4f})")
            print(f"   Worst subject separation: {worst_subject[0]} ({worst_subject[1]:.4f})")
        
        if 'day_scores' in silhouette_results and silhouette_results['day_scores']:
            for day, score in silhouette_results['day_scores'].items():
                print(f"   {day} separation quality: {score:.4f}")
    
    # Create cross-subject combinations
    print(f"\n{'8' if include_silhouette_analysis else '4'}. Creating cross-subject combination plots...")
    plot_cross_subject_combinations(raw3_psd, raw56_psd, save_path)
    
    print("\n=== Analysis Complete ===")
    
    return raw3_psd, raw56_psd, silhouette_results


if __name__ == "__main__":
    # Example usage
    raw3_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/preprocessed/at_least/raw3"
    raw56_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/preprocessed/at_least/raw5&6"
    save_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/visualization_results"
    
    raw3_psd, raw56_psd, silhouette_results = analyze_cross_dataset_distribution(
        raw3_path, raw56_path, save_path, include_silhouette_analysis=True
    )
