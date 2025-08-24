"""
Cross-dataset t-SNE visualization for UNM vs UI data comparison.

This module provides functions to compare the distribution of data between
UNM and UI datasets using t-SNE visualization.
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


def compute_psd(data: np.ndarray, fs: int = 45) -> np.ndarray:
    """
    Computes the Power Spectral Density (PSD) of the data.

    Args:
        data (np.array): The input data
        fs (int): Sampling frequency.

    Returns:
        np.array: The PSD of the data.
    """
    f, Pxx = signal.welch(data, fs=fs)
    return Pxx


def load_unm_ui_dataset_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    Load all .npy files from UNM or UI dataset directory and organize them by subject.
    
    Args:
        data_path (str): Path to the dataset directory (UNM or UI npy folder)
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with subject names as keys and data as values
    """
    psd_data = {}
    npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    
    print(f"Loading data from {data_path}")
    print(f"Found {len(npy_files)} .npy files")
    
    for npy_file in npy_files:
        try:
            data = np.load(os.path.join(data_path, npy_file))
            subject_name = os.path.splitext(npy_file)[0]  # Remove .npy extension
            
            # Compute PSD for the data
            psd = compute_psd(data)
            psd_data[subject_name] = psd
            
            print(f"  Loaded {subject_name}: shape {data.shape} -> PSD shape {psd.shape}")
            
        except Exception as e:
            print(f"  ❌ Error loading {npy_file}: {str(e)}")
            continue
    
    return psd_data


def prepare_combined_data(unm_psd: Dict[str, np.ndarray], ui_psd: Dict[str, np.ndarray]):
    """
    Combine UNM and UI PSD data for t-SNE analysis.
    
    Args:
        unm_psd: Dictionary of UNM PSD data
        ui_psd: Dictionary of UI PSD data
        
    Returns:
        Tuple of (combined_data, dataset_labels, subject_labels, class_labels)
    """
    print("\nPreparing combined data for analysis...")
    
    combined_data = []
    dataset_labels = []  # 0: UNM, 1: UI
    subject_labels = []
    class_labels = []    # 0: CTL, 1: PD
    
    # Add UNM data
    for subject_name, psd_data in unm_psd.items():
        # Each subject may have multiple samples
        for i in range(psd_data.shape[0]):
            combined_data.append(psd_data[i].flatten())
            dataset_labels.append(0)  # UNM
            subject_labels.append(f"UNM_{subject_name}")
            
            # Determine class from subject name
            if subject_name.startswith('CTL'):
                class_labels.append(0)  # CTL
            else:
                class_labels.append(1)  # PD
    
    # Add UI data
    for subject_name, psd_data in ui_psd.items():
        # Each subject may have multiple samples
        for i in range(psd_data.shape[0]):
            combined_data.append(psd_data[i].flatten())
            dataset_labels.append(1)  # UI
            subject_labels.append(f"UI_{subject_name}")
            
            # Determine class from subject name
            if subject_name.startswith('CTL'):
                class_labels.append(0)  # CTL
            else:
                class_labels.append(1)  # PD
    
    combined_data = np.array(combined_data)
    dataset_labels = np.array(dataset_labels)
    subject_labels = np.array(subject_labels)
    class_labels = np.array(class_labels)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"UNM samples: {np.sum(dataset_labels == 0)}")
    print(f"UI samples: {np.sum(dataset_labels == 1)}")
    print(f"CTL samples: {np.sum(class_labels == 0)}")
    print(f"PD samples: {np.sum(class_labels == 1)}")
    
    return combined_data, dataset_labels, subject_labels, class_labels


def plot_comprehensive_comparison(tsne_results, dataset_labels, class_labels, subject_labels, save_path=None):
    """
    Create comprehensive comparison plots showing dataset and class differences.
    """
    print("Creating comprehensive comparison plots...")
    
    # Set up the plot style (matching cross_dataset_tsne.py)
    sns.set_theme(style="white", context="notebook")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Dataset colors and class markers (matching cross_dataset_tsne.py style)
    dataset_colors = {0: '#FF6B6B', 1: '#4ECDC4'}  # UNM: red, UI: teal (same as raw3/raw5&6)
    dataset_names = {0: 'UNM', 1: 'UI'}
    class_colors = {0: '#9CBBE9', 1: '#E6B89D'}  # CTL: blue, PD: orange (same as day1/day8)
    class_names = {0: 'CTL', 1: 'PD'}
    class_markers = {0: 'o', 1: 's'}  # Circle for CTL, Square for PD
    
    # Plot 1: Dataset comparison (top-left)
    ax1 = axes[0, 0]
    for dataset_id in [0, 1]:
        mask = dataset_labels == dataset_id
        ax1.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                   c=dataset_colors[dataset_id], alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5,
                   label=f'{dataset_names[dataset_id]} (n={np.sum(mask)})')
    
    ax1.set_title('Dataset Comparison: UNM vs UI', fontsize=14, fontweight='bold')
    ax1.legend(frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(ax=ax1, left=True, bottom=True)
    
    # Plot 2: Class comparison (top-right)
    ax2 = axes[0, 1]
    for class_id in [0, 1]:
        mask = class_labels == class_id
        ax2.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                   c=class_colors[class_id], alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5,
                   label=f'{class_names[class_id]} (n={np.sum(mask)})')
    
    ax2.set_title('Class Comparison: CTL vs PD', fontsize=14, fontweight='bold')
    ax2.legend(frameon=False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, bottom=True)
    
    # Plot 3: Combined view with dataset colors and class markers (bottom-left)
    ax3 = axes[1, 0]
    for dataset_id in [0, 1]:
        for class_id in [0, 1]:
            mask = (dataset_labels == dataset_id) & (class_labels == class_id)
            if np.sum(mask) > 0:
                ax3.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                           c=dataset_colors[dataset_id], marker=class_markers[class_id],
                           alpha=0.7, s=80, edgecolors='black', linewidth=0.5,
                           label=f'{dataset_names[dataset_id]}_{class_names[class_id]} (n={np.sum(mask)})')
    
    ax3.set_title('Combined View\n(Colors: Dataset, Shapes: Class)', fontsize=14, fontweight='bold')
    ax3.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xticks([])
    ax3.set_yticks([])
    sns.despine(ax=ax3, left=True, bottom=True)
    
    # Plot 4: Subject distribution (bottom-right)
    ax4 = axes[1, 1]
    unique_subjects = np.unique(subject_labels)
    
    # Get unique markers for subjects
    markers_list = ['o', 'X', 's', 'P', '^', 'D', 'v', '<', '>', '*']
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_subjects), 20)))
    
    # Due to many subjects, show only a sample or aggregate view
    for i, subject in enumerate(unique_subjects[:20]):  # Show first 20 subjects
        mask = subject_labels == subject
        if np.sum(mask) > 0:
            dataset_type = 'UNM' if subject.startswith('UNM') else 'UI'
            class_type = 'CTL' if 'CTL' in subject else 'PD'
            marker = markers_list[i % len(markers_list)]
            
            ax4.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                       c=[colors[i % len(colors)]], marker=marker,
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.3,
                       label=f'{subject} (n={np.sum(mask)})')
    
    ax4.set_title(f'Subject Distribution\n(Showing first 20 of {len(unique_subjects)} subjects)', 
                  fontsize=14, fontweight='bold')
    if len(unique_subjects) <= 10:
        ax4.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.set_xticks([])
    ax4.set_yticks([])
    sns.despine(ax=ax4, left=True, bottom=True)
    
    plt.tight_layout()
    
    if save_path:
        # Save comprehensive plot
        plt.savefig(os.path.join(save_path, 'UNM_UI_comprehensive_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_comprehensive_comparison.svg'), 
                    format='svg', bbox_inches='tight')
        print(f"Saved comprehensive comparison plot to {save_path}")
        
        # Save individual plots
        print("Saving individual comparison plots...")
        
        # Save plot 1: Dataset comparison
        fig1, ax1_single = plt.subplots(1, 1, figsize=(8, 6))
        for dataset_id in [0, 1]:
            mask = dataset_labels == dataset_id
            ax1_single.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                              c=dataset_colors[dataset_id], alpha=0.7, s=60,
                              edgecolors='black', linewidth=0.5,
                              label=f'{dataset_names[dataset_id]} (n={np.sum(mask)})')
        
        ax1_single.set_title('Dataset Comparison: UNM vs UI', fontsize=14, fontweight='bold')
        ax1_single.legend(frameon=False)
        ax1_single.set_xticks([])
        ax1_single.set_yticks([])
        sns.despine(ax=ax1_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_1_dataset_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_1_dataset_comparison.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close(fig1)
        
        # Save plot 2: Class comparison
        fig2, ax2_single = plt.subplots(1, 1, figsize=(8, 6))
        for class_id in [0, 1]:
            mask = class_labels == class_id
            ax2_single.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                              c=class_colors[class_id], alpha=0.7, s=60,
                              edgecolors='black', linewidth=0.5,
                              label=f'{class_names[class_id]} (n={np.sum(mask)})')
        
        ax2_single.set_title('Class Comparison: CTL vs PD', fontsize=14, fontweight='bold')
        ax2_single.legend(frameon=False)
        ax2_single.set_xticks([])
        ax2_single.set_yticks([])
        sns.despine(ax=ax2_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_2_class_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_2_class_comparison.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close(fig2)
        
        # Save plot 3: Combined view
        fig3, ax3_single = plt.subplots(1, 1, figsize=(12, 6))
        for dataset_id in [0, 1]:
            for class_id in [0, 1]:
                mask = (dataset_labels == dataset_id) & (class_labels == class_id)
                if np.sum(mask) > 0:
                    ax3_single.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                                      c=dataset_colors[dataset_id], marker=class_markers[class_id],
                                      alpha=0.7, s=80, edgecolors='black', linewidth=0.5,
                                      label=f'{dataset_names[dataset_id]}_{class_names[class_id]} (n={np.sum(mask)})')
        
        ax3_single.set_title('Combined View\n(Colors: Dataset, Shapes: Class)', fontsize=14, fontweight='bold')
        ax3_single.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3_single.set_xticks([])
        ax3_single.set_yticks([])
        sns.despine(ax=ax3_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_3_combined_view.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_3_combined_view.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close(fig3)
        
        # Save plot 4: Subject distribution
        fig4, ax4_single = plt.subplots(1, 1, figsize=(14, 8))
        unique_subjects = np.unique(subject_labels)
        
        # Get unique markers for subjects
        markers_list = ['o', 'X', 's', 'P', '^', 'D', 'v', '<', '>', '*']
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_subjects), 20)))
        
        # Due to many subjects, show only a sample or aggregate view
        for i, subject in enumerate(unique_subjects[:20]):  # Show first 20 subjects
            mask = subject_labels == subject
            if np.sum(mask) > 0:
                dataset_type = 'UNM' if subject.startswith('UNM') else 'UI'
                class_type = 'CTL' if 'CTL' in subject else 'PD'
                marker = markers_list[i % len(markers_list)]
                
                ax4_single.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                                  c=[colors[i % len(colors)]], marker=marker,
                                  alpha=0.7, s=60, edgecolors='black', linewidth=0.3,
                                  label=f'{subject} (n={np.sum(mask)})')
        
        ax4_single.set_title(f'Subject Distribution\n(Showing first 20 of {len(unique_subjects)} subjects)', 
                            fontsize=14, fontweight='bold')
        if len(unique_subjects) <= 10:
            ax4_single.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4_single.set_xticks([])
        ax4_single.set_yticks([])
        sns.despine(ax=ax4_single, left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_4_subject_distribution.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_4_subject_distribution.svg'), 
                    format='svg', bbox_inches='tight')
        plt.close(fig4)
        
        print(f"Saved 4 individual comparison plots to {save_path}")
    
    plt.show()


def compute_silhouette_analysis(tsne_results, dataset_labels, class_labels, subject_labels):
    """
    Compute silhouette analysis for the t-SNE results.
    """
    print("Computing silhouette analysis...")
    
    silhouette_results = {}
    
    # Overall silhouette score based on dataset labels
    overall_score = silhouette_score(tsne_results, dataset_labels)
    silhouette_results['overall_score'] = overall_score
    
    # Silhouette score based on class labels
    class_score = silhouette_score(tsne_results, class_labels)
    silhouette_results['class_score'] = class_score
    
    # Individual sample silhouette scores
    sample_silhouette_values = silhouette_samples(tsne_results, dataset_labels)
    silhouette_results['sample_scores'] = sample_silhouette_values
    
    # Dataset-wise statistics
    unm_scores = sample_silhouette_values[dataset_labels == 0]
    ui_scores = sample_silhouette_values[dataset_labels == 1]
    
    silhouette_results['unm_stats'] = {
        'mean': np.mean(unm_scores),
        'std': np.std(unm_scores),
        'median': np.median(unm_scores)
    }
    
    silhouette_results['ui_stats'] = {
        'mean': np.mean(ui_scores),
        'std': np.std(ui_scores),
        'median': np.median(ui_scores)
    }
    
    print(f"Overall silhouette score (dataset): {overall_score:.4f}")
    print(f"Class silhouette score: {class_score:.4f}")
    print(f"UNM mean±std: {silhouette_results['unm_stats']['mean']:.4f}±{silhouette_results['unm_stats']['std']:.4f}")
    print(f"UI mean±std: {silhouette_results['ui_stats']['mean']:.4f}±{silhouette_results['ui_stats']['std']:.4f}")
    
    return silhouette_results


def plot_silhouette_analysis(tsne_results, dataset_labels, silhouette_results, save_path=None):
    """
    Create comprehensive silhouette analysis visualization with 4 subplots.
    """
    print("Creating silhouette analysis plot...")
    
    # Set up the plot style (matching cross_dataset_tsne.py)
    sns.set_theme(style="white", context="notebook")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    # Color scheme (matching cross_dataset_tsne.py)
    dataset_colors = {0: '#FF6B6B', 1: '#4ECDC4'}  # UNM: red, UI: teal
    dataset_names = {0: 'UNM', 1: 'UI'}
    sample_scores = silhouette_results['sample_scores']
    
    # 1a. UNM t-SNE plot with silhouette scores as color intensity (top-left)
    ax1a = axes[0, 0]
    mask_unm = dataset_labels == 0
    scatter_unm = ax1a.scatter(
        tsne_results[mask_unm, 0], tsne_results[mask_unm, 1],
        c=sample_scores[mask_unm], 
        cmap='viridis',
        alpha=0.7, s=60, 
        edgecolors='black', linewidth=0.3
    )
    ax1a.set_title('UNM t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', 
                   fontsize=14, fontweight='bold')
    ax1a.set_xticks([])
    ax1a.set_yticks([])
    sns.despine(ax=ax1a, left=True, bottom=True)
    plt.colorbar(scatter_unm, ax=ax1a, label='Silhouette Score')
    
    # 1b. UI t-SNE plot with silhouette scores as color intensity (top-middle)
    ax1b = axes[0, 1]
    mask_ui = dataset_labels == 1
    scatter_ui = ax1b.scatter(
        tsne_results[mask_ui, 0], tsne_results[mask_ui, 1],
        c=sample_scores[mask_ui], 
        cmap='viridis',
        alpha=0.7, s=60, 
        edgecolors='black', linewidth=0.3
    )
    ax1b.set_title('UI t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', 
                   fontsize=14, fontweight='bold')
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    sns.despine(ax=ax1b, left=True, bottom=True)
    plt.colorbar(scatter_ui, ax=ax1b, label='Silhouette Score')
    
    # 2a. UNM silhouette score distribution histogram (top-right)
    ax2a = axes[0, 2]
    unm_scores = sample_scores[dataset_labels == 0]
    
    ax2a.hist(unm_scores, bins=30, alpha=0.7, color='#FF6B6B', density=True)
    # ax2a.axvline(np.mean(unm_scores), color='darkred', linestyle='--', 
    #              label=f'UNM Mean: {np.mean(unm_scores):.3f}')
    ax2a.set_xlabel('Silhouette Score')
    ax2a.set_ylabel('Density')
    ax2a.set_title('UNM Silhouette Score Distribution', fontsize=14, fontweight='bold')
    ax2a.legend(frameon=False)
    ax2a.grid(alpha=0.3)
    
    # 2b. UI silhouette score distribution histogram (bottom-left)
    ax2b = axes[1, 0]
    ui_scores = sample_scores[dataset_labels == 1]
    
    ax2b.hist(ui_scores, bins=30, alpha=0.7, color='#4ECDC4', density=True)
    # ax2b.axvline(np.mean(ui_scores), color='darkgreen', linestyle='--', 
    #              label=f'UI Mean: {np.mean(ui_scores):.3f}')
    ax2b.set_xlabel('Silhouette Score')
    ax2b.set_ylabel('Density')
    ax2b.set_title('UI Silhouette Score Distribution', fontsize=14, fontweight='bold')
    ax2b.legend(frameon=False)
    ax2b.grid(alpha=0.3)
    
    # 3. Silhouette plot by dataset (bottom-middle)
    ax3 = axes[1, 1]
    y_lower = 10
    for i in [0, 1]:  # For each dataset
        dataset_silhouette_values = sample_scores[dataset_labels == i]
        dataset_silhouette_values.sort()
        
        size_cluster_i = dataset_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        dataset_name = dataset_names[i]
        color = dataset_colors[i]
        
        ax3.fill_betweenx(np.arange(y_lower, y_upper),
                          0, dataset_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, f'{dataset_name}')
        y_lower = y_upper + 10
    
    ax3.set_xlabel('Silhouette coefficient values')
    ax3.set_ylabel('Dataset labels')
    ax3.set_title(f'Silhouette Analysis\n(Overall Score: {silhouette_results["overall_score"]:.3f})', 
                  fontsize=14, fontweight='bold')
    
    # Vertical line for average silhouette score
    # ax3.axvline(x=silhouette_results['overall_score'], color="red", linestyle="--", alpha=0.8)
    ax3.grid(alpha=0.3)
    
    # 4. Summary statistics comparison (bottom-right)
    ax4 = axes[1, 2]
    categories = ['UNM', 'UI']
    means = [silhouette_results['unm_stats']['mean'], silhouette_results['ui_stats']['mean']]
    stds = [silhouette_results['unm_stats']['std'], silhouette_results['ui_stats']['std']]
    
    bars = ax4.bar(categories, means, yerr=stds, capsize=5, 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Silhouette Score')
    ax4.set_title('Dataset Comparison\n(Mean ± Std)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        # Save comprehensive plot
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_analysis_comprehensive.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_analysis_comprehensive.svg'), 
                    format='svg', bbox_inches='tight')
        print(f"Saved comprehensive silhouette analysis plot to {save_path}")
        
        # Save individual plots
        print("Saving individual plots...")
        
        # Save plot 1a: UNM t-SNE with silhouette scores
        fig1a, ax1a_single = plt.subplots(1, 1, figsize=(8, 6))
        scatter_unm_single = ax1a_single.scatter(
            tsne_results[mask_unm, 0], tsne_results[mask_unm, 1],
            c=sample_scores[mask_unm], 
            cmap='viridis',
            alpha=0.7, s=60, 
            edgecolors='black', linewidth=0.3
        )
        ax1a_single.set_title('UNM t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', 
                              fontsize=14, fontweight='bold')
        ax1a_single.set_xticks([])
        ax1a_single.set_yticks([])
        sns.despine(ax=ax1a_single, left=True, bottom=True)
        plt.colorbar(scatter_unm_single, ax=ax1a_single, label='Silhouette Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_1a_UNM_tsne.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_1a_UNM_tsne.svg'), format='svg', bbox_inches='tight')
        plt.close(fig1a)
        
        # Save plot 1b: UI t-SNE with silhouette scores
        fig1b, ax1b_single = plt.subplots(1, 1, figsize=(8, 6))
        scatter_ui_single = ax1b_single.scatter(
            tsne_results[mask_ui, 0], tsne_results[mask_ui, 1],
            c=sample_scores[mask_ui], 
            cmap='viridis',
            alpha=0.7, s=60, 
            edgecolors='black', linewidth=0.3
        )
        ax1b_single.set_title('UI t-SNE with Silhouette Scores\n(Color intensity = silhouette score)', 
                              fontsize=14, fontweight='bold')
        ax1b_single.set_xticks([])
        ax1b_single.set_yticks([])
        sns.despine(ax=ax1b_single, left=True, bottom=True)
        plt.colorbar(scatter_ui_single, ax=ax1b_single, label='Silhouette Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_1b_UI_tsne.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_1b_UI_tsne.svg'), format='svg', bbox_inches='tight')
        plt.close(fig1b)
        
        # Save plot 2a: UNM histogram
        fig2a, ax2a_single = plt.subplots(1, 1, figsize=(8, 6))
        unm_scores_single = sample_scores[dataset_labels == 0]
        ax2a_single.hist(unm_scores_single, bins=30, alpha=0.7, color='#FF6B6B', density=True)
        # ax2a_single.axvline(np.mean(unm_scores_single), color='darkred', linestyle='--', 
        #                    label=f'UNM Mean: {np.mean(unm_scores_single):.3f}')
        ax2a_single.set_xlabel('Silhouette Score')
        ax2a_single.set_ylabel('Density')
        ax2a_single.set_title('UNM Silhouette Score Distribution', fontsize=14, fontweight='bold')
        ax2a_single.legend(frameon=False)
        ax2a_single.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_2a_UNM_histogram.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_2a_UNM_histogram.svg'), format='svg', bbox_inches='tight')
        plt.close(fig2a)
        
        # Save plot 2b: UI histogram
        fig2b, ax2b_single = plt.subplots(1, 1, figsize=(8, 6))
        ui_scores_single = sample_scores[dataset_labels == 1]
        ax2b_single.hist(ui_scores_single, bins=30, alpha=0.7, color='#4ECDC4', density=True)
        # ax2b_single.axvline(np.mean(ui_scores_single), color='darkgreen', linestyle='--', 
        #                    label=f'UI Mean: {np.mean(ui_scores_single):.3f}')
        ax2b_single.set_xlabel('Silhouette Score')
        ax2b_single.set_ylabel('Density')
        ax2b_single.set_title('UI Silhouette Score Distribution', fontsize=14, fontweight='bold')
        ax2b_single.legend(frameon=False)
        ax2b_single.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_2b_UI_histogram.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_2b_UI_histogram.svg'), format='svg', bbox_inches='tight')
        plt.close(fig2b)
        
        # Save plot 3: Silhouette analysis
        fig3, ax3_single = plt.subplots(1, 1, figsize=(8, 6))
        y_lower = 10
        for i in [0, 1]:  # For each dataset
            dataset_silhouette_values = sample_scores[dataset_labels == i]
            dataset_silhouette_values.sort()
            
            size_cluster_i = dataset_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            dataset_name = dataset_names[i]
            color = dataset_colors[i]
            
            ax3_single.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, dataset_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
            
            ax3_single.text(-0.05, y_lower + 0.5 * size_cluster_i, f'{dataset_name}')
            y_lower = y_upper + 10
        
        ax3_single.set_xlabel('Silhouette coefficient values')
        ax3_single.set_ylabel('Dataset labels')
        ax3_single.set_title(f'Silhouette Analysis\n(Overall Score: {silhouette_results["overall_score"]:.3f})', 
                            fontsize=14, fontweight='bold')
        # ax3_single.axvline(x=silhouette_results['overall_score'], color="red", linestyle="--", alpha=0.8)
        ax3_single.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_3_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_3_analysis.svg'), format='svg', bbox_inches='tight')
        plt.close(fig3)
        
        # Save plot 4: Summary statistics
        fig4, ax4_single = plt.subplots(1, 1, figsize=(8, 6))
        categories = ['UNM', 'UI']
        means = [silhouette_results['unm_stats']['mean'], silhouette_results['ui_stats']['mean']]
        stds = [silhouette_results['unm_stats']['std'], silhouette_results['ui_stats']['std']]
        
        bars = ax4_single.bar(categories, means, yerr=stds, capsize=5, 
                             color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
        ax4_single.set_ylabel('Mean Silhouette Score')
        ax4_single.set_title('Dataset Comparison\n(Mean ± Std)', fontsize=14, fontweight='bold')
        ax4_single.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax4_single.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_4_summary.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'UNM_UI_silhouette_4_summary.svg'), format='svg', bbox_inches='tight')
        plt.close(fig4)
        
        print(f"Saved 6 individual silhouette analysis plots to {save_path}")
    
    plt.show()


def analyze_unm_ui_distribution(
    unm_path: str, 
    ui_path: str, 
    save_path: Optional[str] = None,
    include_silhouette_analysis: bool = True
) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Main function to analyze and visualize UNM vs UI dataset distribution differences.
    
    Args:
        unm_path: Path to UNM dataset npy directory
        ui_path: Path to UI dataset npy directory
        save_path: Optional path to save plots
        include_silhouette_analysis: Whether to include silhouette score analysis
        
    Returns:
        Tuple of (unm_psd_data, ui_psd_data, silhouette_results)
    """
    print("=== UNM vs UI Dataset Distribution Analysis ===")
    print(f"UNM path: {unm_path}")
    print(f"UI path: {ui_path}")
    
    # Create save directory if specified
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"Results will be saved to: {save_path}")
    
    silhouette_results = None
    
    # Load datasets
    print("\n1. Loading UNM dataset...")
    unm_psd = load_unm_ui_dataset_data(unm_path)
    
    print("\n2. Loading UI dataset...")
    ui_psd = load_unm_ui_dataset_data(ui_path)
    
    # Prepare combined data
    print("\n3. Preparing combined data...")
    combined_data, dataset_labels, subject_labels, class_labels = prepare_combined_data(unm_psd, ui_psd)
    
    # Standardize data
    print("\n4. Standardizing data...")
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # Perform t-SNE
    print("\n5. Performing t-SNE analysis...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_data)//4), 
                max_iter=1000, n_jobs=-1)
    tsne_results = tsne.fit_transform(combined_data_scaled)
    
    # Create comprehensive comparison plots
    print("\n6. Creating comprehensive comparison plots...")
    plot_comprehensive_comparison(
        tsne_results=tsne_results,
        dataset_labels=dataset_labels,
        class_labels=class_labels,
        subject_labels=subject_labels,
        save_path=save_path
    )
    
    # Silhouette analysis
    if include_silhouette_analysis:
        print("\n7. Computing silhouette analysis...")
        silhouette_results = compute_silhouette_analysis(
            tsne_results=tsne_results,
            dataset_labels=dataset_labels,
            class_labels=class_labels,
            subject_labels=subject_labels
        )
        
        print("\n8. Creating silhouette score visualization...")
        plot_silhouette_analysis(
            tsne_results=tsne_results,
            dataset_labels=dataset_labels,
            silhouette_results=silhouette_results,
            save_path=save_path
        )
        
        # Display key results summary
        print(f"\n🎯 Key Silhouette Analysis Results:")
        print(f"   Overall Silhouette Score (Dataset): {silhouette_results['overall_score']:.4f}")
        print(f"   Class Silhouette Score: {silhouette_results['class_score']:.4f}")
        print(f"   Total Samples: {len(tsne_results)}")
        print(f"   UNM samples: {np.sum(dataset_labels == 0)}")
        print(f"   UI samples: {np.sum(dataset_labels == 1)}")
        print(f"   CTL samples: {np.sum(class_labels == 0)}")
        print(f"   PD samples: {np.sum(class_labels == 1)}")
    
    print("\n=== Analysis Complete ===")
    
    return unm_psd, ui_psd, silhouette_results


if __name__ == "__main__":
    # Example usage
    unm_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/preprocessed/UNM/npy"
    ui_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/data/preprocessed/UI/npy"
    save_path = "/home/jsw/Fairness/tmp/Fairness_for_generalization/visualization_results"
    
    unm_psd, ui_psd, silhouette_results = analyze_unm_ui_distribution(
        unm_path, ui_path, save_path, include_silhouette_analysis=True
    )
