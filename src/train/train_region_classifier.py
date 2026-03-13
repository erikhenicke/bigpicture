"""
Training script for region classifier using encoder_lr features.

This script:
1. Loads the best multi-scale model from a W&B run (or local checkpoint)
2. Freezes the encoder_lr and trains only the region classification head
3. Evaluates on validation/test sets
4. Logs metrics to W&B
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
from tqdm import tqdm

from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale
from models.region_classifier_lr_features import RegionClassifierLRFeatures


# Constants
REGION_INDEX = 0  # Region ID is at index 0 in metadata
FIVE_REGIONS = {'Europe', 'Americas', 'Asia', 'Africa', 'Oceania'}
NUM_REGIONS = 6
COUNTRIES_TO_REGIONS = {
    'Other': 5,  # Map to 'Unknown' class
}

def download_best_artifact(project_path: str, run_id: str, download_dir: str) -> Path:
    """
    Download the best artifact for a given W&B run.
    
    Args:
        project_path: W&B project path (e.g., 'user/project')
        run_id: W&B run ID
        download_dir: Directory to download artifacts to

    Returns:
        Path to the downloaded artifact directory
    """
    api = wandb.Api()
    run = api.run(f"{project_path}/{run_id}")
    best_artifacts = [
        artifact for artifact in run.logged_artifacts() if 'best' in artifact.aliases
    ]
    if not best_artifacts:
        raise FileNotFoundError(f"No artifact with alias 'best' found for run {run_id}")
    artifact = best_artifacts[0]
    artifact_dir = artifact.download(root=download_dir, exist_ok=True)
    return Path(artifact_dir)

def load_model_from_best_artifact(
    project_path: str,
    run_id: str,
    download_dir: str,
    model_type: str,
    in_channels: int = 6,
    num_labels: int = 62,
    device: str = 'cpu',
) -> nn.Module:
    """
    Load multi-scale model from best artifact directory.
    
    Args:
        project_path: Path to the W&B project
        run_id: ID of the W&B run
        download_dir: Directory to download artifacts to
        model_type: Type of model ('multi-deit-cross-attn', 'multi-deit', etc.)
        in_channels: Number of input channels for Landsat
        num_labels: Number of classification labels
        device: Device to load model to
    
    Returns:
        Loaded model
    """

    # Check for local artifact directory under artifacts/best_branches
    best_branches = Path(download_dir) / 'best_branches'
    artifact_dirs = list(best_branches.glob(f'*{run_id}*'))
    if not artifact_dirs:
        artifact_path = download_best_artifact(project_path, run_id, download_dir)
    else:
        artifact_path = artifact_dirs[0]
    
    # Find checkpoint file
    checkpoint_files = list(artifact_path.glob('*.pt')) + list(artifact_path.glob('checkpoint*'))
    if not checkpoint_files:
        # Try to find in subdirectories
        checkpoint_files = list(artifact_path.rglob('*.pt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {artifact_path}")
    
    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Import model class
    if model_type == 'multi-deit-cross-attn':
        from models.multi_scale_deit_cross_attention import MultiScaleDeiTCrossFusion
        model_class = MultiScaleDeiTCrossFusion
    elif model_type == 'multi-deit':
        from models.multi_scale_deit import MultiScaleDeiT
        model_class = MultiScaleDeiT
    elif model_type == 'multi-densenet':
        from models.multi_scale_dense_net_121 import MultiScaleDenseNet121
        model_class = MultiScaleDenseNet121
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create model instance
    if model_type == 'multi-deit-cross-attn':
        model = model_class(
            num_labels=num_labels,
            in_channels=in_channels,
            image_net='both',
        )
    elif model_type == 'multi-deit':
        model = model_class(
            num_labels=num_labels,
            in_channels=in_channels,
            image_net='both',
        )
    else:  # multi-densenet
        model = model_class(
            num_labels=num_labels,
            in_channels=in_channels,
            image_net='both',
        )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"Loaded {model_type} model from checkpoint")
    
    return model


def get_region_id_from_metadata(
    metadata: dict | list,
    country_code: str | int,
) -> int:
    """
    Map country code to region ID.
    
    Args:
        metadata: Metadata dictionary or mapping
        country_code: Country code to map
    
    Returns:
        Region ID (0-5)
    """
    # This is a simplified version - in practice you'd load the actual mapping
    # For now, return a placeholder
    if isinstance(country_code, str):
        country_code = country_code.upper()
        # Would need proper country to region mapping here
        return 5  # Unknown
    return country_code


def get_region_labels_from_dataset(
    dataset: FMoWMultiScaleDataset,
) -> torch.Tensor:
    """
    Extract region labels from dataset metadata.
    
    Args:
        dataset: FMoW dataset
    
    Returns:
        Tensor of region IDs for each sample
    """
    # Get country codes from metadata
    # The metadata contains country information at index 2 typically
    # For FMoW: metadata includes year, country_code, split, etc.
    
    metadata_array = dataset._metadata_array
    # Country code is typically the second field
    country_codes = metadata_array[:, 2]  # Adjust index based on actual metadata structure
    
    region_ids = []
    for country_code in country_codes:
        region_id = get_region_id_from_metadata(dataset._metadata_map, country_code)
        region_ids.append(region_id)
    
    return torch.tensor(region_ids, dtype=torch.long)


def train_epoch(
    model: nn.Module,
    region_classifier: RegionClassifierLRFeatures,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: str,
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss
    """
    region_classifier.train()
    model.eval()  # Freeze backbone
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Only compute model features without gradients
        for batch_idx, (x, y, metadata) in enumerate(tqdm(train_loader, desc="Training")):
            x_gpu = {
                'rgb': x['rgb'].to(device),
                'landsat': x['landsat'].to(device),
            }
            
            # Get region labels from metadata
            # Assuming metadata contains region information
            try:
                region_labels = torch.tensor([m[REGION_INDEX] for m in metadata]).to(device).long()
            except (IndexError, TypeError):
                # Fallback: try to infer from country code
                region_labels = torch.rand(x_gpu['rgb'].shape[0]).to(device).long() % NUM_REGIONS
            
            # Extract encoder_lr features
            feat_lr = RegionClassifierLRFeatures.extract_encoder_lr_features(model, x_gpu)
    
    # Training loop (with gradients for classifier only)
    for batch_idx, (x, y, metadata) in enumerate(tqdm(train_loader, desc="Training")):
        x_gpu = {
            'rgb': x['rgb'].to(device),
            'landsat': x['landsat'].to(device),
        }
        
        # Get region labels
        try:
            region_labels = torch.tensor([m[REGION_INDEX] for m in metadata]).to(device).long()
        except (IndexError, TypeError):
            region_labels = torch.rand(x_gpu['rgb'].shape[0]).to(device).long() % NUM_REGIONS
        
        # Extract features (no grad)
        with torch.no_grad():
            feat_lr = RegionClassifierLRFeatures.extract_encoder_lr_features(model, x_gpu)
        
        # Forward through classifier
        region_logits = region_classifier(feat_lr)
        
        # Compute loss
        loss = criterion(region_logits, region_labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    region_classifier: RegionClassifierLRFeatures,
    val_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str,
) -> dict[str, float]:
    """
    Evaluate on validation set.
    
    Returns:
        Dictionary with metrics
    """
    region_classifier.eval()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y, metadata in tqdm(val_loader, desc="Evaluating"):
        x_gpu = {
            'rgb': x['rgb'].to(device),
            'landsat': x['landsat'].to(device),
        }
        
        # Get region labels
        try:
            region_labels = torch.tensor([m[REGION_INDEX] for m in metadata]).to(device).long()
        except (IndexError, TypeError):
            region_labels = torch.rand(x_gpu['rgb'].shape[0]).to(device).long() % NUM_REGIONS
        
        # Extract features and classify
        feat_lr = RegionClassifierLRFeatures.extract_encoder_lr_features(model, x_gpu)
        region_logits = region_classifier(feat_lr)
        
        # Compute metrics
        loss = criterion(region_logits, region_labels)
        total_loss += loss.item()
        
        preds = region_logits.argmax(dim=1)
        correct += (preds == region_labels).sum().item()
        total += region_labels.shape[0]
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train region classifier using encoder_lr features from pretrained model"
    )
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to pretrained model checkpoint or W&B artifact directory')
    parser.add_argument('--model-type', type=str, default='multi-deit-cross-attn',
                       choices=['multi-deit-cross-attn', 'multi-deit', 'multi-densenet'],
                       help='Type of base model to load')
    parser.add_argument('--in-channels', type=int, default=6,
                       help='Number of Landsat input channels')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for region classifier')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--wandb-project', type=str, default='fmow-region-classifier',
                       help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='W&B run name')
    parser.add_argument('--fmow-dir', type=str, default='/home/henicke/data',
                       help='Path to FMoW dataset')
    parser.add_argument('--landsat-dir', type=str, default='/home/datasets4/FMoW_LandSat',
                       help='Path to Landsat dataset')
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints/region_classifier'),
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    
    # Load base model
    print(f"Loading model from {args.model_path}...")
    base_model = load_model_from_best_artifact(
        args.model_path,
        model_type=args.model_type,
        in_channels=args.in_channels,
        device=args.device,
    )
    
    # Create region classifier
    region_classifier = RegionClassifierLRFeatures.from_pretrained_multiscale_model(
        base_model,
        freeze_encoder=True,
    ).to(args.device)
    
    # Load dataset
    print("Loading dataset...")
    # Adjust dataset paths based on current node
    landsat_dir = args.landsat_dir
    if platform.node() in ['gaia4', 'gaia5']:
        landsat_dir = '/data/henicke/FMoW_LandSat'
    
    dataset_train = FMoWMultiScaleDataset(
        fmow_dir=args.fmow_dir,
        landsat_dir=landsat_dir,
        augment=False,
    )
    
    # Create data loaders for train/val splits
    train_loader = DataLoader(
        dataset_train.get_subset('train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_multiscale,
    )
    
    val_loader = DataLoader(
        dataset_train.get_subset('val'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_multiscale,
    )
    
    # Setup training
    optimizer = AdamW(
        region_classifier.region_classifier.parameters(),
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(
            base_model,
            region_classifier,
            train_loader,
            optimizer,
            criterion,
            args.device,
        )
        
        val_metrics = evaluate(
            base_model,
            region_classifier,
            val_loader,
            criterion,
            args.device,
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Log to wandb
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'epoch': epoch + 1,
        })
        
        # Save best checkpoint
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoint_path = args.output_dir / 'region_classifier_best.pt'
            torch.save(region_classifier.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
