import os
import random
import numpy as np
from tqdm import tqdm
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from torchmetrics.functional import accuracy
import time
import argparse
import json
import logging
from typing import Dict, Any, Optional

# Import enhanced project utilities
# from utils.project_setup import project_paths, model_registry, VersionCompatibility, ConfigValidator
from src.data.providers.dataProvider import DataProvider
from pytorch_lightning.loggers import TensorBoardLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_reproducibility(seed: int) -> None:
    """Setup reproducible random state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)
    logger.info(f"Reproducibility setup with seed: {seed}")


def get_model_class(model_name: str) -> type:
    """Get model class using automatic discovery."""
    model_class = model_registry.get_model(model_name)
    if model_class is None:
        available_models = model_registry.list_models()
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )
    logger.info(f"Using model: {model_name}")
    return model_class


def validate_configuration(data_config: Dict[str, Any]) -> None:
    """Validate configuration before training."""
    config_errors = ConfigValidator.validate_data_config(data_config)
    if config_errors:
        logger.error(f"Configuration validation failed: {config_errors}")
        raise ValueError(f"Invalid configuration: {', '.join(config_errors)}")
    logger.info("Configuration validation passed")


def main(args):
    """Main training function with enhanced setup and validation."""
    # Check version compatibility
    VersionCompatibility.check_pytorch_lightning()
    
    # Set up reproducibility
    setup_reproducibility(args.seed)
    
    # Get model class using automatic discovery
    model_class = get_model_class(args.model_name)

    # Hyperparameters and configurations
    masking_ch_list = args.masking_ch_list
    rm_ch_list = args.rm_ch_list
    fold_k = args.fold_k
    class_weight = args.class_weight
    train_time = time.strftime("%y%m%d_%H%M%S")
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    # Load and validate configuration
    try:
        with open(args.data_config, 'r') as f:
            data_config = json.load(f)
        validate_configuration(data_config)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load data configuration: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Instantiate the DataProvider
    data_provider = DataProvider(data_config, masking_ch_list, rm_ch_list, args.subject_usage)
    dataset = data_provider.get_dataset()
    chans, samples = data_provider.get_data_shape()
    logger.info(f"Data shape - Channels: {chans}, Samples: {samples}")
    
    # Trainer logger
    tb_logger = TensorBoardLogger('tb_logs', name=f'EEGNet_{train_time}')

    # K-Fold Cross-Validation
    kf = KFold(n_splits=fold_k, shuffle=True, random_state=args.seed)
    logger.info(f"Starting {fold_k}-fold cross-validation")
    
    # Training Loop with PyTorch Lightning
    scores_list = []
    acc_list = []
    f1s_list = []
    fold_no = 1
    
    for train_index, test_index in tqdm(kf.split(dataset)):
        print("\n" + "⚡"*50 + "\n")
        print(f'Fold {fold_no}')        
    
        # Split data indices
        train_indices = train_index
        test_indices = test_index
    
        # Further split train_indices into train and val
        val_samples = int(len(train_indices) * 0.1)
        val_indices = train_indices[:val_samples]
        train_indices = train_indices[val_samples:]
    
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
    
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
        # Initialize the model with error handling
        try:
            model_kwargs = {
                'nb_classes': args.nb_classes,
                'Chans': chans,
                'Samples': samples,
                'kernLength': 256,
                'class_weight': class_weight
            }
            
            if args.checkpoint_path is not None:
                model_kwargs['checkpoint_path'] = args.checkpoint_path
                logger.info(f"Loading model from checkpoint: {args.checkpoint_path}")
            
            model = model_class(**model_kwargs)
            logger.info(f"Model initialized: {model.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
        # Log model parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters - Trainable: {trainable_params:,}, Total: {total_params:,}")

        # Define checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath='./model_save/checkpoint/',
            filename=f'checkpoint_{train_time}_fold{fold_no}',
            save_top_k=1,
            verbose=True,
            monitor=args.monitor_value_name,
            mode='min'
        )
    
        # Initialize the Trainer with enhanced configuration
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=args.gpu if torch.cuda.is_available() else None,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            callbacks=[checkpoint_callback],
            deterministic=True,
            logger=tb_logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            # Uncomment for debugging
            # limit_train_batches=0.1,
            # limit_val_batches=0.1,
        )

        # Log data split information
        logger.info(f"Fold {fold_no} - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        logger.info(f"Train set data shape: {data_provider.data_x[train_indices].shape}, {data_provider.data_y[train_indices].shape}")
        logger.info(f"Validation set data shape: {data_provider.data_x[val_indices].shape}, {data_provider.data_y[val_indices].shape}")
        logger.info(f"Test set data shape: {data_provider.data_x[test_indices].shape}, {data_provider.data_y[test_indices].shape}")
        
        data_provider.print_sample()
        print("\n" + "⚡"*50 + "\n")


        # Train the model with error handling
        try:
            logger.info(f"Starting training for fold {fold_no}")
            trainer.fit(model, train_loader, val_loader)
            logger.info(f"Training completed for fold {fold_no}")
        except Exception as e:
            logger.error(f"Training failed for fold {fold_no}: {e}")
            raise
    
        # Test the model
        try:
            logger.info(f"Starting testing for fold {fold_no}")
            test_results = trainer.test(model, test_loader)
            logger.info(f"Testing completed for fold {fold_no}")
        except Exception as e:
            logger.error(f"Testing failed for fold {fold_no}: {e}")
            raise
        # test_results = trainer.callback_metrics
        # scores_list.append(test_results)
    
        # Get predictions and compute metrics
        test_metrics = test_results[0]
        domain_list = data_config.get('domain_list')
        acc = test_metrics['test_acc' if domain_list is None else 'test_target_acc']
        f1s = test_metrics['test_f1' if domain_list is None else 'test_target_f1']
        acc_list.append(acc)
        f1s_list.append(f1s)
        scores_list.append(test_metrics)

        print(f'Test Accuracy for fold {fold_no}: {acc}')
        print(f'F1 Score for fold {fold_no}: {f1s}')

        # Save test data
        test_data_save_path = f"./model_save/test_data/{train_time}"
        os.makedirs(test_data_save_path, exist_ok=True)

        np.save(f"{test_data_save_path}/{train_time}_fold{fold_no}_x.npy", data_provider.data_x[test_indices])
        np.save(f"{test_data_save_path}/{train_time}_fold{fold_no}_y.npy", data_provider.data_y[test_indices])

        fold_no += 1
        # preds = []
        # trues = []
        # model.eval()
        # with torch.no_grad():
        #     for batch in test_loader:
        #         x, y = batch
        #         x = x.cuda() if torch.cuda.is_available() else x
        #         y = y.cuda() if torch.cuda.is_available() else y
        #         logits = model(x)
        #         pred = torch.argmax(logits, dim=1)
        #         preds.extend(pred.cpu().numpy())
        #         trues.extend(y.cpu().numpy())
    
        # acc = np.mean(np.array(preds) == np.array(trues))
        # acc_list.append(acc)
    
        # f1s = f1_score(trues, preds, average='binary')
        # f1s_list.append(f1s)
    
        # print(f'Test Accuracy for fold {fold_no}: {acc}')
        # print(f'F1 Score for fold {fold_no}: {f1s}')
    
        # # Save test data
        # test_data_save_path = f"./model_save/test_data/{train_time}"
        # os.makedirs(test_data_save_path, exist_ok=True)
    
        # np.save(f"{test_data_save_path}/{train_time}_fold{fold_no}_x.npy", data_provider.data_x[test_indices])
        # np.save(f"{test_data_save_path}/{train_time}_fold{fold_no}_y.npy", data_provider.data_y[test_indices])

    # Print overall results
    print('Cross-Validation Results:')
    print(f'Average Accuracy: {np.mean(acc_list)}')
    print(f'Average F1 Score: {np.mean(f1s_list)}')


    model_info_save_path = "./model_save/model_info/"
    os.makedirs(model_info_save_path, exist_ok=True)

    # Write model information to a text file
    with open(f"{model_info_save_path}{train_time}.txt", "w") as f:
        detail = ""
        detail += f"\nModel created time: {train_time}"
        detail += f"\n\nModel explain: {args.model_explain}"

        data_list = data_config['data_list']

        # Log data paths used in training
        for typ in data_list.keys():
            for lb in data_list[typ].keys():
                for d in data_list[typ][lb]:
                    detail += f"\n{typ}\t/{lb}\t/{d}"

        detail += f"\n\nScores List: {scores_list}"
        detail += f"\n\nAccuracy list: {acc_list}"
        detail += f"\nAccuracy average: {np.mean(acc_list)}"
        detail += f"\n\nF1 score list: {f1s_list}"
        detail += f"\nF1 score average: {np.mean(f1s_list)}"

        f.write(detail)
    print(f"Model information saved to {model_info_save_path}{train_time}.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEGNet model with PyTorch Lightning')

    # Add arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--masking_ch_list', nargs='+', type=int, default=[], help='List of channels to mask')
    parser.add_argument('--rm_ch_list', nargs='+', type=int, default=[16, 17], help='List of channels to remove')
    parser.add_argument('--class_weight', nargs='+', type=int, default=[1,1], help='Class weight for class 0')
    parser.add_argument('--fold_k', type=int, default=5, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--window_size', type=int, default=6, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--nb_classes', type=int, default=2, help='target class number')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--data_config', type=str, required=True, help='Path to JSON file containing data paths and configurations: data_list, skip_list')
    parser.add_argument('--gpu', type=int, default=1 ,required=False, help='usage gpu num, 1<=gpu')
    parser.add_argument('--model_explain', type=str, required=True, help='parameter for write model info')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--monitor_value_name', type=str, required=True, help='monitor value name')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='model checkpoint path')
    parser.add_argument('--subject_usage', type=str, required=False, default="all",help='use all subject to train or except one subject to test? : all, test1')

    args = parser.parse_args()
    main(args)