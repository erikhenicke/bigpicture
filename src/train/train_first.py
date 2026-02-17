from models.single_scale_deit import SingleScaleDeiT
from models.multi_scale_deit import MultiResolutionDeiT
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb


def get_data(frac: float = 0.1):
    dataset = FMoWMultiScaleDataset(
        fmow_dir='/home/henicke/git/bigpicture/data',
        landsat_dir='/home/datasets4/FMoW_LandSat',
    )
    return dataset.get_subset(split='train', frac=frac), dataset.get_subset(split='val', frac=frac)

def get_loader(data, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_multiscale):
    return DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def make(config: dict, device='cuda'):
    train_data, val_data = get_data(frac=config.frac)
    train_loader, val_loader = get_loader(train_data, batch_size=config.batch_size), get_loader(val_data, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize model
    print(f"Initializing {config.model_type} model...")
    if config.model_type == 'single':
        model = SingleScaleDeiT(num_labels=62)
    else:
        model = MultiResolutionDeiT(num_labels=62)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = CrossEntropyLoss()

    return model, train_loader, val_loader, optimizer, criterion

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

def evaluate_batch(x, y, model, criterion, device):
    model.eval()

    x = {k: v.to(device) for k, v in x.items()}
    y = y.to(device)

    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

    return loss.item(), correct / total

def train_log(loss, acc, sample_ct, epoch):
    wandb.log({'train_loss': loss, 'train_acc': acc, 'epoch': epoch}, step=sample_ct)

def val_log(loss, acc, epoch):
    wandb.log({'val_loss': loss, 'val_acc': acc}, step=epoch)

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

def evaluate_epoch(model, dataloader, criterion, device, epoch):
    val_loss = 0
    val_correct = 0
    val_total = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        x, y, _ = batch
        loss, acc = evaluate_batch(x, y, model, criterion, device)
        val_loss += loss
        val_correct += acc * y.size(0)
        val_total += y.size(0)
    
    # Log validation metrics after each epoch
    val_loss /= len(dataloader)
    val_acc = val_correct / val_total
    val_log(val_loss, val_acc, epoch)


def train(model, train_loader, val_loader, optimizer, criterion, device, config):

    wandb.watch(model, criterion, log='all', log_freq=10)
    
    sample_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        sample_ct, batch_ct = train_epoch(model, train_loader, optimizer, criterion, device, sample_ct, batch_ct, epoch)
        evaluate_epoch(model, val_loader, criterion, device, epoch)


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
        model, train_loader, val_loader, optimizer, criterion = make(config, device)

        train(model, train_loader, val_loader, optimizer, criterion, device, config)

    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frac', type=float, default=0.01, help='Fraction of data to use')
    args = parser.parse_args()

    wandb.login()
    
    print("="*50)
    print(f"Running {args.model_type.upper()} model experiment")
    print("="*50)
    
    model = run_experiment(args)
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
