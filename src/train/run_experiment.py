from models.single_scale_deit import SingleScaleDeiT
from models.multi_scale_deit import MultiScaleDeiT
from models.single_scale_dense_net_121 import SingleScaleDenseNet121 
from models.multi_scale_dense_net_121 import MultiScaleDenseNet121
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale

import platform
from collections import defaultdict
import tempfile

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
import wandb

# To compute per-region metrics, the region-id is retrieved from sample metadata at index 0 (_metadata_array in FMoWDataset class).
REGION_INDEX = 0
FIVE_REGIONS = {"Europe", "Americas", "Asia", "Africa", "Oceania"}
NUM_CLASSES = 62
SGD_MOMENTUM = 0.9
DATA_LOADER_COLLATE_FN = collate_multiscale
DATA_LOADER_NUM_WORKERS = 4


def get_data_loader(data: torch.utils.data.Dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=DATA_LOADER_NUM_WORKERS,
        collate_fn=DATA_LOADER_COLLATE_FN
    )


def get_data_loaders(eval_splits: list, batch_size: int, frac: float):
    fmow_dir = '/home/henicke/data'
    landsat_dir = '/home/datasets4/FMoW_LandSat'
    preprocessed_dir = None

    if platform.node() == 'gaia4' or platform.node() == 'gaia5':
        preprocessed_dir = '/data/henicke/FMoW_LandSat'

    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir
    )
    train_dataset = dataset.get_subset(split='train', frac=frac)
    eval_datasets = [dataset.get_subset(split=split, frac=frac) for split in eval_splits]
    return get_data_loader(train_dataset, batch_size=batch_size, shuffle=True), *[get_data_loader(data, batch_size=batch_size, shuffle=False) for data in eval_datasets]


def make(config: dict, device='cuda'):
    train_loader, val_od_loader, val_id_loader, test_od_loader, test_id_loader = get_data_loaders(
        eval_splits=['val', 'id_val', 'test', 'id_test'], 
        batch_size=config.batch_size, 
        frac=config.frac
    )
    train_data = train_loader.dataset
    val_od_data = val_od_loader.dataset
    val_id_data = val_id_loader.dataset
    test_od_data = test_od_loader.dataset
    test_id_data = test_id_loader.dataset
    
    print(
        f"Train size: {len(train_data)}, "
        f"Val-OD size: {len(val_od_data)}, "
        f"Val-ID size: {len(val_id_data)}, "
        f"Test-OD size: {len(test_od_data)}, "
        f"Test-ID size: {len(test_id_data)}"
    )

    data_loaders = (train_loader, val_od_loader, val_id_loader, test_od_loader, test_id_loader)
    
    # Initialize model
    print(f"Initializing {config.model_type} model...")
    if config.model_type == 'single-deit':
        model = SingleScaleDeiT(num_labels=NUM_CLASSES)
    elif config.model_type == 'multi-deit':
        model = MultiScaleDeiT(num_labels=NUM_CLASSES, in_channels=config.landsat_in_channels)
    elif config.model_type == 'single-dense-net-121':
        model = SingleScaleDenseNet121(num_labels=NUM_CLASSES)
    else:
        model = MultiScaleDenseNet121(num_labels=NUM_CLASSES, in_channels=config.landsat_in_channels)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    if config.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=SGD_MOMENTUM)

    criterion = CrossEntropyLoss()

    return model, data_loaders, optimizer, criterion

def train_batch(x, y, model, optimizer, criterion, device):
    model.train()

    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    preds = torch.argmax(logits, dim=1)
    correct = (preds == y).sum().item()
    total = y.size(0)

    return loss.item(), correct / total

def train_log(loss, acc, epoch):
    wandb.log({'train_loss': loss, 'train_acc': acc}, step=epoch)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    for batch_ct, batch in enumerate(tqdm(dataloader, desc="Training")):
        x, y, _ = batch
        loss, acc = train_batch(x, y, model, optimizer, criterion, device)

        # Log to TensorBoard and Weights & Biases
        if batch_ct % 25 == 0:
            epoch_progress = epoch + (batch_ct / len(dataloader))
            train_log(loss, acc, epoch_progress)

def _get_region_names(dataloader):
    dataset = dataloader.dataset
    base_dataset = getattr(dataset, "dataset", dataset)
    metadata_map = getattr(base_dataset, "_metadata_map", None)
    if metadata_map and "region" in metadata_map:
        return list(metadata_map["region"])
    return []

def evaluate_batch(x, y, metadata, model, criterion, device, calibration_metric=None):
    model.eval()

    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)
    region_ids = metadata[:, REGION_INDEX].to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        correct_mask = (preds == y)
        correct = correct_mask.sum().item()
        entropy = -(probs * torch.log(probs.clamp_min(0e-12))).sum(dim=1)
        entropy_correct = entropy[correct_mask].sum().item()
        entropy_incorrect = entropy[~correct_mask].sum().item()

        if calibration_metric is not None:
            calibration_metric.update(probs, y)

    region_correct = defaultdict(int)
    region_total = defaultdict(int)
    for rid in torch.unique(region_ids):
        rid_int = rid.item()
        mask = region_ids == rid
        region_total[rid_int] += mask.sum().item()
        region_correct[rid_int] += (correct_mask & mask).sum().item()

    return (
        loss.item(),
        correct,
        region_correct,
        region_total,
        entropy_correct,
        entropy_incorrect,
    )

def val_log(prefix, loss, acc, epoch, extra_metrics=None):
    payload = {f'{prefix}-loss': loss, f'{prefix}-acc': acc}
    if extra_metrics:
        payload.update(extra_metrics)
    wandb.log(payload, step=epoch)


def log_model_artifact(model, artifact_name, aliases=None, metadata=None):
    artifact = wandb.Artifact(artifact_name, type='model')
    if metadata:
        artifact.metadata.update(metadata)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)
        artifact.add_file(tmp_file.name, name='model.pt')
    wandb.log_artifact(artifact, aliases=aliases)

# TODO: Validate
def delete_old_artifact_versions(artifact_name: str, keep_alias: str):
    if wandb.run is None:
        return
    api = wandb.Api()
    full_name = f"{wandb.run.entity}/{wandb.run.project}/{artifact_name}"
    versions = api.artifact_versions("model", full_name)
    for version in versions:
        if keep_alias not in version.aliases:
            version.delete(delete_aliases=True)

def evaluate(model, dataloader, criterion, device, data_prefix, ece_n_bins=15):
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_entropy_correct = 0
    val_entropy_incorrect = 0 
    region_correct_total = defaultdict(int)
    region_total_total = defaultdict(int)
    region_names = _get_region_names(dataloader)
    ece_metric = MulticlassCalibrationError(
        num_classes=NUM_CLASSES,
        n_bins=ece_n_bins,
        norm='l1'
    ).to(device)

    for batch in tqdm(dataloader, desc="Evaluating"):
        x, y, metadata = batch
        (
            loss,
            correct,
            region_correct,
            region_total,
            entropy_correct,
            entropy_incorrect,
        ) = evaluate_batch(
            x, y, metadata, model, criterion, device, calibration_metric=ece_metric
        )
        val_loss += loss
        val_entropy_correct += entropy_correct
        val_entropy_incorrect += entropy_incorrect 
        val_correct += correct
        val_total += y.size(0)

        for rid, count in region_total.items():
            region_total_total[rid] += count
        for rid, count in region_correct.items():
            region_correct_total[rid] += count
    
    # Log validation metrics after each epoch
    val_loss /= len(dataloader)
    val_acc = val_correct / val_total
    val_ece = ece_metric.compute().item()
    val_entropy_correct /= val_correct if val_correct > 0 else 1
    val_entropy_incorrect /= (val_total - val_correct) if (val_total - val_correct) > 0 else 1

    per_region_acc = {}
    for rid, total in region_total_total.items():
        if total == 0:
            continue
        name = region_names[rid] if rid < len(region_names) else str(rid)
        per_region_acc[name] = region_correct_total[rid] / total

    filtered_acc = [acc for name, acc in per_region_acc.items() if name in FIVE_REGIONS]
    worst_group_acc = min(filtered_acc) if filtered_acc else 0.0

    metrics = {
        f"{data_prefix}-loss": val_loss,
        f"{data_prefix}-acc": val_acc,
        f"{data_prefix}-worst-group-acc": worst_group_acc,
        f"{data_prefix}-ece": val_ece,
        f"{data_prefix}-entropy-correct": val_entropy_correct,
        f"{data_prefix}-entropy-incorrect": val_entropy_incorrect,
    }
    for name, acc in per_region_acc.items():
        metrics[f"{data_prefix}-region-{name.lower()}-acc"] = acc

    return metrics 


def train(model, data_loaders, optimizer, criterion, device, config):
    train_loader, val_od_loader, val_id_loader, test_od_loader, test_id_loader = data_loaders
    # Group validation and test loaders
    val_loaders = {'val-od': val_od_loader, 'val-id': val_id_loader}
    test_loaders = {'test-od': test_od_loader, 'test-id': test_id_loader}
    
    best_worst_group_acc = 0
    
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        if config.learning_rate_decay < 1.0:
            lr = config.learning_rate * (config.learning_rate_decay ** epoch)
            optimizer.param_groups[0]['lr'] = lr
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Evaluate on validation splits
        for prefix, loader in val_loaders.items():
            eval_metrics = evaluate(
                model,
                loader,
                criterion,
                device,
                epoch,
                data_prefix=prefix,
                ece_n_bins=config.ece_n_bins,
            )
            # TODO: Log validation metrics to Weights & Biases
            
            # Track best worst group accuracy on val-od split
            if prefix == 'val-od' and eval_metrics[f"{prefix}-worst-group-acc"] > best_worst_group_acc:
                best_worst_group_acc = eval_metrics[f"{prefix}-worst-group-acc"]
                # TODO: Eval best model on test
                log_model_artifact(
                    model,
                    artifact_name='best-model',
                    aliases=['best'],
                    metadata={
                        'worst_group_acc': best_worst_group_acc,
                        'epoch': epoch,
                    },
                )
                delete_old_artifact_versions('best-model', keep_alias='best')
                print(f"Logged best model with worst-group-acc: {best_worst_group_acc:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_model_artifact(
                model,
                artifact_name=f'checkpoint-epoch-{epoch}',
                metadata={'epoch': epoch},
            )
            print(f"Logged checkpoint at epoch {epoch}")
    
    # Save final model
    log_model_artifact(
        model,
        artifact_name='final-model',
        aliases=['final'],
        metadata={'total_epochs': config.epochs},
    )
    print("Logged final model")
    
    # Evaluate on test splits
    final_epoch = config.epochs
    for prefix, loader in test_loaders.items():
        evaluate(
            model,
            loader,
            criterion,
            device,
            final_epoch,
            data_prefix=prefix,
            ece_n_bins=config.ece_n_bins,
        )


def run_experiment(args):
    """
    Run experiment with specified model type.
    
    Args:
        args: Contains 
            model_type: 'single' or 'multi'
            num_epochs: Number of training epochs
            batch_size: Batch size
            frac: Fraction of data to use (for quick experiments, use 0.1)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    with wandb.init(project='fmow-deit', config=args):
        config = wandb.config
        model, data_loaders, optimizer, criterion = make(config, device)

        train(model, data_loaders, optimizer, criterion, device, config)

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='single', choices=['single-deit', 'multi-deit', 'single-dense-net-121', 'multi-dense-net-121'])
    parser.add_argument('--landsat_in_channels', type=int, default=6, help='Number of input channels for Landsat data (default: 6)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--learning_rate_decay', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ece_n_bins', type=int, default=10, help='Number of bins for ECE (Expected Calibration Error)')
    args = parser.parse_args()

    wandb.login()
    
    print("="*50)
    print(f"Running {args.model_type.upper()} model experiment")
    print("="*50)
    
    model = run_experiment(args)
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
