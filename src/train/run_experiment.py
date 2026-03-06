from models.single_scale_deit import SingleScaleDeiT
from models.single_scale_deit_landsat import SingleScaleDeiTLandsat
from models.multi_scale_deit import MultiScaleDeiT
from models.single_scale_dense_net_121 import SingleScaleDenseNet121 
from models.multi_scale_dense_net_121 import MultiScaleDenseNet121
from models.multi_scale_deit_cross_attention import MultiScaleDeiTCrossFusion
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale

import platform
from collections import defaultdict
import tempfile

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
import wandb

# To compute per-region metrics, the region-id is retrieved from sample metadata at index 0 (_metadata_array in FMoWDataset class).
REGION_INDEX = 0
FIVE_REGIONS = {'Europe', 'Americas', 'Asia', 'Africa', 'Oceania'}
NUM_REGIONS = 6
NUM_CLASSES = 62
SGD_MOMENTUM = 0.9
DATA_LOADER_COLLATE_FN = collate_multiscale
DATA_LOADER_NUM_WORKERS = 4


def get_data_loader(dataset: FMoWMultiScaleDataset, split, config, shuffle=False):
    return DataLoader(
        dataset.get_subset(split, frac=config.frac), 
        batch_size=config.batch_size, 
        shuffle=shuffle, 
        num_workers=DATA_LOADER_NUM_WORKERS,
        collate_fn=DATA_LOADER_COLLATE_FN
    )


def make_data_loaders(train_split, val_splits, test_splits, speaking_names, config):
    fmow_dir = '/home/henicke/data'
    landsat_dir = '/home/datasets4/FMoW_LandSat'
    preprocessed_dir = None

    if platform.node() == 'gaia4' or platform.node() == 'gaia5':
        preprocessed_dir = '/data/henicke/FMoW_LandSat'
        
    dataset_augment = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir,
        augment=config.data_augmentation
    )

    dataset = FMoWMultiScaleDataset(
        fmow_dir=fmow_dir,
        landsat_dir=landsat_dir,
        preprocessed_dir=preprocessed_dir
    )

    return (
        get_data_loader(dataset_augment, train_split, config, shuffle=True), 
        {speaking_names[split]: get_data_loader(dataset, split, config, shuffle=False) for split in val_splits},
        {speaking_names[split]: get_data_loader(dataset, split, config, shuffle=False) for split in test_splits}
    )

def make_model(config: dict, device: str):
    print(f'Initializing {config.model_type} model...')
    if config.model_type == 'single-deit':
        model = SingleScaleDeiT(num_labels=NUM_CLASSES, image_net=(config.image_net != 'none'))
    elif config.model_type == 'single-deit-landsat':
        model = SingleScaleDeiTLandsat(
            num_labels=NUM_CLASSES,
            in_channels=config.landsat_in_channels,
            image_net=(config.image_net != 'none'),
        )
    elif config.model_type == 'multi-deit':
        model = MultiScaleDeiT(num_labels=NUM_CLASSES, in_channels=config.landsat_in_channels, image_net=config.image_net)
    elif config.model_type == 'multi-deit-cross-attn':
        model = MultiScaleDeiTCrossFusion(
            num_labels=NUM_CLASSES, 
            in_channels=config.landsat_in_channels, 
            image_net=config.image_net, 
            cross_attn_depths=config.cross_attention_depths,
            num_regions=NUM_REGIONS,
            region_aux_enabled=config.region_aux_enabled,
        )
    elif config.model_type == 'single-densenet':
        model = SingleScaleDenseNet121(num_labels=NUM_CLASSES, image_net=(config.image_net != 'none'))
    else:
        model = MultiScaleDenseNet121(num_labels=NUM_CLASSES, in_channels=config.landsat_in_channels, image_net=config.image_net)
    
    return model.to(device)

def make_optimizer(model, config):
    # For multi-deit-cross-attn with region aux, create separate optimizers
    if config.model_type == 'multi-deit-cross-attn' and config.region_aux_enabled:
        # Main optimizer: backbone + fusion + main classifier (exclude region_classifier)
        main_params = []
        for name, p in model.named_parameters():
            if not name.startswith('region_classifier.'):
                main_params.append(p)
        
        if config.optimizer == 'adamw':
            optimizer_main = AdamW(main_params, lr=config.learning_rate, weight_decay=config.weight_decay)
            optimizer_region = AdamW(model.region_classifier.parameters(), lr=config.region_aux_lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'adam':
            optimizer_main = Adam(main_params, lr=config.learning_rate, weight_decay=config.weight_decay)
            optimizer_region = Adam(model.region_classifier.parameters(), lr=config.region_aux_lr, weight_decay=config.weight_decay)
        else:
            optimizer_main = SGD(main_params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=SGD_MOMENTUM)
            optimizer_region = SGD(model.region_classifier.parameters(), lr=config.region_aux_lr, weight_decay=config.weight_decay, momentum=SGD_MOMENTUM)
        
        optimizer = {'main': optimizer_main, 'region': optimizer_region}
        
        scheduler = None
        if config.learning_rate_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer_main,
                mode='max',
                factor=config.learning_rate_decay,
                patience=config.plateau_patience,
            )
        elif config.learning_rate_scheduler == 'step':
            scheduler = StepLR(optimizer_main, step_size=1, gamma=config.learning_rate_decay)
    else:
        # Standard single optimizer
        if config.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=SGD_MOMENTUM)
        
        scheduler = None
        if config.learning_rate_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.learning_rate_decay,
                patience=config.plateau_patience,
            )
        elif config.learning_rate_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=1, gamma=config.learning_rate_decay)

    return optimizer, scheduler

def print_setup_summary(model, data_loaders, optimizer, criterion):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal of model parameters: {total_params:,}')
    print('\nData loaders:')
    for name, loader in data_loaders.items():
        print(f'{name}: {len(loader.dataset)} samples')
    
    if isinstance(optimizer, dict):
        print(f'\nOptimizer (main): {optimizer["main"].__class__.__name__}')
        print(f'Optimizer (region): {optimizer["region"].__class__.__name__}')
    else:
        print(f'\nOptimizer: {optimizer.__class__.__name__}')
    print(f'Criterion: {criterion.__class__.__name__}')

def make_training_context(config: dict, device: str):
    val_splits = ['val', 'id_val']
    test_splits = ['test', 'id_test']
    speaking_names = {'train': 'train', 'val': 'val-od', 'id_val': 'val-id', 'test': 'test-od', 'id_test': 'test-id'}

    train_loader, val_loaders, test_loaders = make_data_loaders(train_split='train', val_splits=val_splits, test_splits=test_splits, speaking_names=speaking_names, config=config)
    model = make_model(config, device)
    optimizer, scheduler = make_optimizer(model, config)
    criterion = CrossEntropyLoss()
    
    # Separate criterion for region classification
    criterion_region = CrossEntropyLoss() if config.region_aux_enabled else None

    print_setup_summary(model, {'train': train_loader, **val_loaders, **test_loaders}, optimizer, criterion)

    return {
        'model': model,
        'train_loader': train_loader,
        'val_loaders': val_loaders,
        'test_loaders': test_loaders,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'criterion_region': criterion_region,
        'device': device,
        'config': config,
    }

def train_batch(x, y, metadata, context):
    model = context['model']
    optimizer = context['optimizer']
    criterion = context['criterion']
    criterion_region = context.get('criterion_region')
    device = context['device']
    config = context['config']

    model.train()

    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)
    
    # Check if region aux is enabled for this model
    region_aux_enabled = config.model_type == 'multi-deit-cross-attn' and config.region_aux_enabled
    
    if region_aux_enabled:
        # Two separate backward passes
        region_y = metadata[:, REGION_INDEX].to(device)
        
        # Forward pass (single forward, returns both logits)
        outputs = model(x, return_aux=True)
        class_logits = outputs['class_logits']
        region_logits = outputs['region_logits']
        
        # Backward pass 1: Main classification task
        optimizer['main'].zero_grad()
        loss_main = criterion(class_logits, y)
        loss_main.backward()
        optimizer['main'].step()
        
        # Backward pass 2: Region classification task (only updates region_classifier)
        optimizer['region'].zero_grad()
        loss_region = criterion_region(region_logits, region_y)
        loss_region.backward()
        optimizer['region'].step()
        
        # Compute metrics
        preds = torch.argmax(class_logits, dim=1)
        correct = (preds == y).sum().item()
        
        region_preds = torch.argmax(region_logits, dim=1)
        region_correct = (region_preds == region_y).sum().item()
        
        total = y.size(0)
        
        return {
            'loss': loss_main.item(),
            'acc': correct / total,
            'region_loss': loss_region.item(),
            'region_acc': region_correct / total,
        }
    else:
        # Standard single backward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

        return {'loss': loss.item(), 'acc': correct / total}

def train_log(metrics, epoch_progress):
    log_dict = {
        'train_loss': metrics['loss'], 
        'train_acc': metrics['acc'], 
        'epoch': epoch_progress
    }
    if 'region_loss' in metrics:
        log_dict['train_region_loss'] = metrics['region_loss']
        log_dict['train_region_acc'] = metrics['region_acc']
    wandb.log(log_dict)

def train_epoch(context, epoch):
    train_loader = context['train_loader']
    for batch_ct, batch in enumerate(tqdm(train_loader, desc='Training')):
        x, y, metadata = batch
        metrics = train_batch(x, y, metadata, context)

        # Log to Weights & Biases
        if batch_ct % 25 == 0:
            epoch_progress = epoch + (batch_ct / len(train_loader))
            train_log(metrics, epoch_progress)

def _get_region_names(dataloader):
    dataset = dataloader.dataset
    base_dataset = getattr(dataset, 'dataset', dataset)
    metadata_map = getattr(base_dataset, '_metadata_map', None)
    if metadata_map and 'region' in metadata_map:
        return list(metadata_map['region'])
    return []

def evaluate_batch(x, y, metadata, context, calibration_metric=None):
    model = context['model']
    criterion = context['criterion']
    criterion_region = context.get('criterion_region')
    device = context['device']
    config = context['config']

    model.eval()
    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)
    region_ids = metadata[:, REGION_INDEX].to(device)
    
    region_aux_enabled = config.model_type == 'multi-deit-cross-attn' and config.region_aux_enabled

    with torch.no_grad():
        if region_aux_enabled:
            outputs = model(x, return_aux=True)
            logits = outputs['class_logits']
            region_logits = outputs['region_logits']
            
            # Region classification metrics
            loss_region = criterion_region(region_logits, region_ids)
            region_preds = torch.argmax(region_logits, dim=1)
            region_correct_count = (region_preds == region_ids).sum().item()
        else:
            logits = model(x)
            loss_region = None
            region_correct_count = 0
            
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
        loss_region.item() if loss_region is not None else 0.0,
        region_correct_count,
    )


def log_model_artifact(model, artifact_name, metadata, optimizer, epoch, config, aliases=None, save_full_checkpoint=False):
    artifact = wandb.Artifact(artifact_name, type='model')
    if metadata:
        artifact.metadata.update(metadata)

    if save_full_checkpoint:
        # Handle dict optimizer (region aux) or single optimizer
        if isinstance(optimizer, dict):
            optimizer_state = {
                'main': optimizer['main'].state_dict(),
                'region': optimizer['region'].state_dict(),
            }
        else:
            optimizer_state = optimizer.state_dict()
            
        payload = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'epoch': epoch,
            'config': dict(config),
        }
        file_name = 'checkpoint.pt'
    else:
        payload = model.state_dict()
        file_name = 'model_state_dict.pt'

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as tmp_file:
        torch.save(payload, tmp_file.name)
        artifact.add_file(tmp_file.name, name=file_name)
    wandb.log_artifact(artifact, aliases=aliases)


def evaluate(loader, loader_name, context, config):
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_entropy_correct = 0
    val_entropy_incorrect = 0 
    val_region_loss = 0
    val_region_correct = 0
    region_correct_total = defaultdict(int)
    region_total_total = defaultdict(int)
    region_names = _get_region_names(loader)
    ece_metric = MulticlassCalibrationError(
        num_classes=NUM_CLASSES,
        n_bins=config.ece_n_bins,
        norm='l1'
    ).to(context['device'])

    for batch in tqdm(loader, desc='Evaluating'):
        x, y, metadata = batch
        (
            loss,
            correct,
            region_correct,
            region_total,
            entropy_correct,
            entropy_incorrect,
            region_loss,
            region_correct_count,
        ) = evaluate_batch(
            x, y, metadata, context, calibration_metric=ece_metric
        )
        val_loss += loss
        val_entropy_correct += entropy_correct
        val_entropy_incorrect += entropy_incorrect 
        val_correct += correct
        val_total += y.size(0)
        val_region_loss += region_loss
        val_region_correct += region_correct_count

        for rid, count in region_total.items():
            region_total_total[rid] += count
        for rid, count in region_correct.items():
            region_correct_total[rid] += count
    
    val_loss /= len(loader)
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
        f'{loader_name}-loss': val_loss,
        f'{loader_name}-acc': val_acc,
        f'{loader_name}-worst-group-acc': worst_group_acc,
        f'{loader_name}-ece': val_ece,
        f'{loader_name}-entropy-correct': val_entropy_correct,
        f'{loader_name}-entropy-incorrect': val_entropy_incorrect,
    }
    
    # Add region classification metrics if enabled
    region_aux_enabled = config.model_type == 'multi-deit-cross-attn' and config.region_aux_enabled
    if region_aux_enabled:
        val_region_loss /= len(loader)
        val_region_acc = val_region_correct / val_total
        metrics[f'{loader_name}-region-loss'] = val_region_loss
        metrics[f'{loader_name}-region-acc'] = val_region_acc
    
    for name, acc in per_region_acc.items():
        metrics[f'{loader_name}-region-{name.lower()}-acc'] = acc

    return metrics 


def evaluate_loaders(loaders, context, config):
    metrics = {} 
    for loader_name, loader in loaders.items():
        metrics.update(evaluate(
        loader,
        loader_name,
        context,
        config,
        ))
    return metrics

def train(context, config):
    
    best_worst_group_acc = 0
    scheduler = context['scheduler']
    optimizer = context['optimizer']
    if wandb.run is None:
        raise RuntimeError('Expected an active wandb run before training starts.')
    run_artifact_name = f'run-{wandb.run.name}'

    for epoch in tqdm(range(config.epochs), desc='Epochs'):
        train_epoch(context, epoch)

        val_metrics = evaluate_loaders(context['val_loaders'], context, config)
        val_metrics.update({'epoch': epoch + 1})

        if scheduler is not None:
            if config.learning_rate_scheduler == 'plateau':
                scheduler.step(val_metrics['val-od-worst-group-acc'])
            else:
                scheduler.step()

        wandb.log(val_metrics)

        is_best = False
        # Track best worst group accuracy on val-od split
        if val_metrics['val-od-worst-group-acc'] > best_worst_group_acc:
            is_best = True
            best_worst_group_acc = val_metrics['val-od-worst-group-acc']
            test_metrics = evaluate_loaders(context['test_loaders'], context, config) 

            log_model_artifact(
                context['model'],
                artifact_name=run_artifact_name,
                aliases=['best', f'checkpoint-epoch-{epoch+1}'],
                metadata={
                    **val_metrics, 
                    **test_metrics, 
                    'epoch': epoch+1,
                    'checkpoint_type': 'best',
                },
                optimizer=optimizer,
                epoch=epoch+1,
                config=config,
            )
            print(f'Logged best model with worst-group-acc: {best_worst_group_acc:.4f}')
        
        # Save checkpoint every 5 epochs
        if not is_best and (epoch + 1) % 5 == 0:
            log_model_artifact(
                context['model'],
                artifact_name=run_artifact_name,
                aliases=[f'checkpoint-epoch-{epoch+1}'],
                metadata={'epoch': epoch+1, 'checkpoint_type': 'periodic'},
                optimizer=optimizer,
                epoch=epoch,
                config=config,
            )
            print(f'Logged checkpoint at epoch {epoch + 1}')
    
    # Save final model
    log_model_artifact(
        context['model'],
        artifact_name=run_artifact_name,
        aliases=['final'],
        metadata={'total_epochs': config.epochs, 'checkpoint_type': 'final'},
        optimizer=optimizer,
        epoch=config.epochs,
        config=config,
        save_full_checkpoint=True,
    )
    print('Logged final model')


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
    print(f'Using device: {device}')

    run_name = f'{args.model_type}-init-{args.image_net}-{args.optimizer}-{args.learning_rate:.0e}-{datetime.now().strftime("%m-%d")}'.replace('.', '_')

    with wandb.init(project='fmow', name=run_name, config=args):
        config = wandb.config
        training_context = make_training_context(config, device)
        train(training_context, config)

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='single-deit', choices=['single-deit', 'single-deit-landsat', 'multi-deit', 'multi-deit-cross-attn', 'single-densenet', 'multi-densenet'])
    parser.add_argument('--image_net', type=str, default='both', choices=['both', 'hr', 'none'], help='Whether to initialize multi-scale branches with ImageNet pre-trained weights')
    parser.add_argument('--landsat_in_channels', type=int, default=6, help='Number of input channels for Landsat data (default: 6)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--learning_rate_scheduler', type=str, default='none', choices=['none', 'step', 'plateau'])
    parser.add_argument('--learning_rate_decay', type=float, default=1.0)
    parser.add_argument('--plateau_patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ece_n_bins', type=int, default=10, help='Number of bins for ECE (Expected Calibration Error)')
    parser.add_argument('--data_augmentation', action='store_true', default=False, help='Whether to apply data augmentation (random horizontal and vertical flips)')
    parser.add_argument('--cross_attention_depths', type=int, nargs='+', default=None, help='List of layer indices at which to apply cross-attention (only for multi-deit-cross-attn model)')
    parser.add_argument('--region_aux_enabled', action='store_true', default=False, help='Enable auxiliary region classification task on LR encoder CLS token (multi-deit-cross-attn only)')
    parser.add_argument('--region_aux_lr', type=float, default=1e-4, help='Learning rate for region classifier head (default: 1e-3)')
    args = parser.parse_args()

    wandb.login()
    
    print('='*50)
    print(f'Running {args.model_type.upper()} model experiment')
    print('='*50)
    
    run_experiment(args)
    
    print('\n' + '='*50)
    print('EXPERIMENT COMPLETE')
    print('='*50)
