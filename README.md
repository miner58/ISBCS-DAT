# ISBCS-DAT: Cross-Subject and Cross-Modality Generalization in ECoG-Based Parkinson's Disease Detection

**Cross-Subject and Cross-Modality Generalization in ECoG-Based Parkinson's Disease Detection via Counterfactual Data Augmentation and Adversarial Training**

This repository contains the implementation of ISBCS-DAT, a novel approach for robust ECoG-based Parkinson's disease detection that generalizes across subjects and modalities using Distribution Mismatch Robust Representation (DMMR) learning.

## Key Features

- **Cross-Subject Generalization**: DMMR-based leave-one-subject-out validation across multiple datasets
- **Cross-Modality Adaptation**: Wireless ↔ Wired ECoG domain adaptation using Gradient Reversal Layer (GRL)
- **Counterfactual Data Augmentation**: Advanced EEG signal augmentation for robustness
- **Adversarial Training**: Domain-invariant feature learning for cross-dataset generalization

## Datasets

- **UI Dataset**: Wireless EEG (14 subjects, real-world conditions)
- **UNM Dataset**: Wired EEG (25 subjects, laboratory conditions)
- **ECoG Dataset**: Electrocorticography (wireless ↔ wire cross-modality experiments)
- **Raw3/Raw5&6**: Mouse EEG datasets for validation

## Quick Start

```bash
# Install dependencies
pip install lightning ray[tune] torchmetrics scikit-learn

# Run DMMR cross-subject experiment
python src/training/hyperparameter_tuning/runner_raytune.py --config src/config/raytune_config/RayTune/UI/DMMR_test.yml

# Run cross-modality experiment
python src/training/hyperparameter_tuning/runner_raytune.py --config src/config/raytune_config/RayTune/UI/DMMR_allMouse.yml
```

## Architecture

```
src/
├── models/dmmr/           # DMMR implementation
├── data/modules/          # EEG data processing
├── training/              # Training pipelines
├── utils/                 # Analysis tools
└── config/                # Experiment configurations
```

## Citation

```bibtex
@article{isbcs_dat_2025,
  title={ISBCS-DAT: Cross-Subject and Cross-Modality Generalization in ECoG-Based Parkinson's Disease Detection via Counterfactual Data Augmentation and Adversarial Training},
  author={[Your Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License.