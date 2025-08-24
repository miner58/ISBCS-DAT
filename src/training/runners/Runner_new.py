import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import time
import sys
import argparse
import json

# Corrected import statements
# Use project setup utilities
try:
    from src.utils.project_setup import project_paths
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

# ✅ Updated to use available models instead of missing GU models
from src.models.eegnet import EEGNet
from src.models.eegnet_grl import EEGNetGRL as EEGNetDomainAdaptation
from src.data.modules.EEGdataModuel import EEGDataModule

def main(args):
    # Set the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)
    
    # Select model
    model_dict = {
        'EEGNet': EEGNet,
        'EEGNetDomainAdaptation': EEGNetDomainAdaptation,
    }
    model_class = model_dict[args.model_name]

    # Hyperparameters and configurations
    masking_ch_list = args.masking_ch_list
    rm_ch_list = args.rm_ch_list
    class_weight = args.class_weight
    train_time = time.strftime("%y%m%d_%H%M%S")
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    with open(args.data_config, 'r') as f:
        data_config = json.load(f)

    # Instantiate the DataModule
    data_module = EEGDataModule(
        data_config=data_config,
        batch_size=batch_size,
        masking_ch_list=masking_ch_list,
        rm_ch_list=rm_ch_list,
        subject_usage=args.subject_usage,
        seed=seed
    )

    # Setup data module
    data_module.setup('fit')
    data_module.setup('test')

    # Get number of channels and samples from data module
    chans = data_module.chans
    samples = data_module.samples
    print(f"Channels (Chans): {chans}")
    print(f"Samples per channel (Samples): {samples}")

    output_path = os.path.join('./model_save'+args.comment+'_'+args.subject_name)

    # Trainer logger
    logger = TensorBoardLogger(os.path.join(output_path, 'tb_log'), name=f'EEGNet_{train_time}')

    # Initialize the model
    if args.checkpoint_path is not None:
        model = model_class(
            nb_classes=args.nb_classes,
            Chans=chans,
            Samples=samples,
            kernLength=256,
            class_weight=class_weight,
            checkpoint_path=args.checkpoint_path,
            grl_lambda=args.grl_lambda
        )
    else:
        model = model_class(
            nb_classes=args.nb_classes,
            Chans=chans,
            Samples=samples,
            kernLength=256,
            class_weight=class_weight,
            grl_lambda=args.grl_lambda
        )

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_path, 'checkpoint'),
        filename=f'checkpoint_{train_time}',
        save_top_k=1,
        verbose=True,
        monitor=args.monitor_value_name,
        mode='min'
    )

    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[args.gpu] if torch.cuda.is_available() else None,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        deterministic=True,
        logger=logger,
        enable_progress_bar=False
        # Uncomment the lines below for debugging purposes
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
    )

    # Print dataset sizes
    print(f"Train set size: {len(data_module.train_dataset)}")
    print(f"Validation set size: {len(data_module.val_dataset)}")
    print(f"Test set size: {len(data_module.test_dataset)}")
    print("\n" + "⚡"*50 + "\n")


    if args.run_pre_test:
        print("Running preliminary test before training...")
        preliminary_test_results = trainer.test(model, datamodule=data_module)
        preliminary_test_metrics = preliminary_test_results[0]
        domain_list = data_config.get('domain_list')
        pre_acc = preliminary_test_metrics['test_acc' if domain_list is None else 'test_target_acc']
        pre_f1s = preliminary_test_metrics['test_f1' if domain_list is None else 'test_target_f1']
        print(f'Preliminary Test Accuracy: {pre_acc}')
        print(f'Preliminary F1 Score: {pre_f1s}')

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')

    # Get predictions and compute metrics
    test_metrics = test_results[0]
    domain_list = data_config.get('domain_list')
    acc = test_metrics['test_acc' if domain_list is None else 'test_target_acc']
    f1s = test_metrics['test_f1' if domain_list is None else 'test_target_f1']

    print(f'Test Accuracy: {acc}')
    print(f'F1 Score: {f1s}')

    # Save test data
    test_data_save_path = os.path.join(output_path, f'test_data/{train_time}')
    os.makedirs(test_data_save_path, exist_ok=True)

    # Extract test data for saving
    test_dataset = data_module.test_dataset
    test_data_x = [test_dataset[i][0].numpy() for i in range(len(test_dataset))]
    test_data_y = [test_dataset[i][1].item() if isinstance(test_dataset[i][1], torch.Tensor) else test_dataset[i][1] for i in range(len(test_dataset))]

    np.save(f"{test_data_save_path}/{train_time}_test_x.npy", test_data_x)
    np.save(f"{test_data_save_path}/{train_time}_test_y.npy", test_data_y)

    # Print overall results
    print('Training and Testing Completed')
    print(f'Final Test Accuracy: {acc}')
    print(f'Final F1 Score: {f1s}')

    model_info_save_path = os.path.join(output_path, 'model_info')
    os.makedirs(model_info_save_path, exist_ok=True)

    # Write model information to a text file
    with open(f"{model_info_save_path}{train_time}.txt", "w") as f:
        detail = ""
        detail += f"\nModel created time: {train_time}"
        detail += f"\n\nModel explanation: {args.model_explain}"

        data_list = data_config['data_list']

        # Log data paths used in training
        for typ in data_list.keys():
            for lb in data_list[typ].keys():
                for d in data_list[typ][lb]:
                    detail += f"\n{typ}\t/{lb}\t/{d}"

        detail += f"\n\nTest Metrics: {test_metrics}"
        detail += f"\n\nFinal Test Accuracy: {acc}"
        detail += f"\nFinal F1 Score: {f1s}"

        f.write(detail)
    print(f"Model information saved to {model_info_save_path}{train_time}.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEGNet model with PyTorch Lightning')

    # Add arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--masking_ch_list', nargs='+', type=int, default=[], help='List of channels to mask')
    parser.add_argument('--rm_ch_list', nargs='+', type=int, default=[9,16, 17], help='List of channels to remove')
    parser.add_argument('--class_weight', nargs='+', type=float, default=[1.0, 1.0], help='Class weights')
    parser.add_argument('--nb_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--fold_k', type=int, default=5, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--grl_lambda', type=float, default=0.1, help='Number of folds for K-Fold cross-validation')
    parser.add_argument('--data_config', type=str, required=True, help='Path to JSON file containing data paths and configurations')
    parser.add_argument('--gpu', type=int, default=0, required=False, help='GPU device index to use (if any)')
    parser.add_argument('--model_explain', type=str, required=True, help='Description of the model for logging purposes')
    parser.add_argument('--model_name', type=str, required=True, choices=['EEGNet', 'EEGNetDomainAdaptation'], help='Name of the model to use')
    parser.add_argument('--monitor_value_name', type=str, default='val_loss', help='Metric name to monitor for checkpointing')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='Path to model checkpoint')
    parser.add_argument('--subject_usage', type=str, required=False, default="all",help='use all subject to train or except one subject to test? : all, test1')
    parser.add_argument('--comment', type=str, required=False, default="",help='comment')
    parser.add_argument('--subject_name', type=str, required=False, default="",help='test subject name')
    parser.add_argument('--run_pre_test', action='store_true', required=False, default=True, help='Run test before training as a training step')

    args = parser.parse_args()
    
    # Define output path based on argument
    train_time = time.strftime("%y%m%d_%H%M%S")
    output_path = os.path.join('./model_save'+args.comment+'_'+args.subject_name)
    os.makedirs(output_path, exist_ok=True)  # Ensure the output path exists

    # Set up logging to file within output path
    log_filename = os.path.join(output_path, f'log_{train_time}.txt')
    with open(log_filename, 'w') as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        main(args)
