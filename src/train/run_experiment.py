from models.single_scale_deit import SingleScaleDeiT
from models.multi_scale_deit import MultiScaleDeiT
from models.single_scale_dense_net_121 import SingleScaleDenseNet121 
from models.multi_scale_dense_net_121 import MultiScaleDenseNet121
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale

import platform
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
import wandb

REGION_INDEX = 0
FIVE_REGIONS = {"Europe", "Americas", "Asia", "Africa", "Oceania"}
NUM_CLASSES = 62


def get_data(frac: float = 0.1):
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
    return (
        dataset.get_subset(split='train', frac=frac),
        dataset.get_subset(split='val', frac=frac),
        dataset.get_subset(split='id_val', frac=frac),
    )

def get_loader(data, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_multiscale):
    return DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def make(config: dict, device='cuda'):
    train_data, val_od_data, val_id_data = get_data(frac=config.frac)
    train_loader = get_loader(train_data, batch_size=config.batch_size)
    val_od_loader = get_loader(val_od_data, batch_size=config.batch_size, shuffle=False)
    val_id_loader = get_loader(val_id_data, batch_size=config.batch_size, shuffle=False)
    
    print(
        f"Train size: {len(train_data)}, "
        f"Val-OD size: {len(val_od_data)}, "
        f"Val-ID size: {len(val_id_data)}"
    )
    
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
        optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    criterion = CrossEntropyLoss()

    return model, train_loader, val_od_loader, val_id_loader, optimizer, criterion

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

def train_log(loss, acc, sample_ct, epoch):
    wandb.log({'train_loss': loss, 'train_acc': acc, 'epoch': epoch}, step=sample_ct)

def train_epoch(model, dataloader, optimizer, criterion, device, sample_ct, batch_ct, epoch):
    for batch in tqdm(dataloader, desc="Training"):
        x, y, _ = batch
        loss, acc = train_batch(x, y, model, optimizer, criterion, device)
        
        sample_ct += y.size(0)
        batch_ct += 1

        # Log to TensorBoard and Weights & Biases
        if (batch_ct + 1) % 25 == 0:
            train_log(loss, acc, sample_ct, epoch)

    return sample_ct, batch_ct

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
        total = y.size(0)
        entropy = -(probs * torch.log(probs.clamp_min(0e-12))).sum(dim=1)
        entropy_correct = entropy[correct_mask].sum().item()
        entropy_incorrect = entropy[~correct_mask].sum().item()

        if calibration_metric is not None:
            calibration_metric.update(probs, y)

    region_correct = defaultdict(int)
    region_total = defaultdict(int)
    for rid in torch.unique(region_ids):
        rid_int = int(rid.item())
        mask = region_ids == rid
        region_total[rid_int] += int(mask.sum().item())
        region_correct[rid_int] += int((correct_mask & mask).sum().item())

    return (
        loss.item(),
        correct / total,
        region_correct,
        region_total,
        entropy_correct,
        entropy_incorrect,
    )

def val_log(prefix, loss, acc, sample_ct, extra_metrics=None):
    payload = {f'{prefix}-loss': loss, f'{prefix}-acc': acc}
    if extra_metrics:
        payload.update(extra_metrics)
    wandb.log(payload, step=sample_ct)

def evaluate_epoch(model, dataloader, criterion, device, sample_ct, metric_prefix, mce_n_bins=15):
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_entropy_correct = 0
    val_entropy_incorrect = 0 
    region_correct_total = defaultdict(int)
    region_total_total = defaultdict(int)
    region_names = _get_region_names(dataloader)
    mce_metric = MulticlassCalibrationError(
        num_classes=NUM_CLASSES,
        n_bins=mce_n_bins,
        norm='l1'
    ).to(device)

    for batch in tqdm(dataloader, desc="Evaluating"):
        x, y, metadata = batch
        (
            loss,
            acc,
            region_correct,
            region_total,
            entropy_correct,
            entropy_incorrect,
        ) = evaluate_batch(
            x, y, metadata, model, criterion, device, calibration_metric=mce_metric
        )
        val_loss += loss
        val_entropy_correct += entropy_correct
        val_entropy_incorrect += entropy_incorrect 
        val_correct += acc * y.size(0)
        val_total += y.size(0)

        for rid, count in region_total.items():
            region_total_total[rid] += count
        for rid, count in region_correct.items():
            region_correct_total[rid] += count
    
    # Log validation metrics after each epoch
    val_loss /= len(dataloader)
    val_acc = val_correct / val_total
    val_mce = mce_metric.compute().item()
    val_entropy_correct *= val_acc
    val_entropy_incorrect *= (1 - val_acc) 

    per_region_acc = {}
    for rid, total in region_total_total.items():
        if total == 0:
            continue
        name = region_names[rid] if rid < len(region_names) else str(rid)
        per_region_acc[name] = region_correct_total[rid] / total

    filtered_acc = [acc for name, acc in per_region_acc.items() if name in FIVE_REGIONS]
    worst_group_acc = min(filtered_acc) if filtered_acc else 0.0

    extra_metrics = {
        f"{metric_prefix}-worst-group-acc": worst_group_acc,
        f"{metric_prefix}-mce": val_mce,
        f"{metric_prefix}-entropy-correct": val_entropy_correct,
        f"{metric_prefix}-entropy-incorrect": val_entropy_incorrect,
    }
    for name, acc in per_region_acc.items():
        extra_metrics[f"{metric_prefix}-region-{name}-acc"] = acc

    val_log(metric_prefix, val_loss, val_acc, sample_ct, extra_metrics=extra_metrics)


def train(model, train_loader, val_od_loader, val_id_loader, optimizer, criterion, device, config):

    wandb.watch(model, criterion, log='all', log_freq=10)
    
    sample_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        if config.learning_rate_decay < 1.0:
            lr = config.learning_rate * (config.learning_rate_decay ** epoch)
            optimizer.param_groups[0]['lr'] = lr
        sample_ct, batch_ct = train_epoch(model, train_loader, optimizer, criterion, device, sample_ct, batch_ct, epoch)
        evaluate_epoch(
            model,
            val_od_loader,
            criterion,
            device,
            sample_ct,
            metric_prefix='val-od',
            mce_n_bins=config.mce_n_bins,
        )
        evaluate_epoch(
            model,
            val_id_loader,
            criterion,
            device,
            sample_ct,
            metric_prefix='val-id',
            mce_n_bins=config.mce_n_bins,
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
        model, train_loader, val_od_loader, val_id_loader, optimizer, criterion = make(config, device)

        train(model, train_loader, val_od_loader, val_id_loader, optimizer, criterion, device, config)

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
    parser.add_argument('--mce_n_bins', type=int, default=10, help='Number of bins for Multiclass Calibration Error')
    args = parser.parse_args()

    wandb.login()
    
    print("="*50)
    print(f"Running {args.model_type.upper()} model experiment")
    print("="*50)
    
    model = run_experiment(args)
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
