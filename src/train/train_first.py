from models.single_scale_deit import SingleScaleDeiT
from models.multi_scale_deit import MultiResolutionDeiT
from dataset.fmow_multiscale_dataset import FMoWMultiScaleDataset, collate_multiscale

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        x_dict, y, metadata = batch
        
        # Move to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_dict)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x_dict, y, metadata = batch
            
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            y = y.to(device)
            
            logits = model(x_dict)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def run_experiment(model_type='single', num_epochs=10, batch_size=32, frac=1.0):
    """
    Run experiment with specified model type.
    
    Args:
        model_type: 'single' or 'multi'
        num_epochs: Number of training epochs
        batch_size: Batch size
        frac: Fraction of data to use (for quick experiments, use 0.1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = FMoWMultiScaleDataset(
        root_dir='data',
        fmow_rgb_dir='images',
        landsat_dir='landsat_images',
        split_scheme='official'
    )
    
    # Get train/val subsets
    print(f"Creating subsets (using {frac*100}% of data)...")
    train_data = dataset.get_subset(split='train', frac=frac)
    val_data = dataset.get_subset(split='val', frac=frac)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_multiscale
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_multiscale
    )
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    if model_type == 'single':
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
    
    # Training loop
    best_val_acc = 0
    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    return results, best_val_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frac', type=float, default=0.1, help='Fraction of data to use')
    args = parser.parse_args()
    
    print("="*50)
    print(f"Running {args.model.upper()} model experiment")
    print("="*50)
    
    results, best_acc = run_experiment(
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        frac=args.frac
    )
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print(f"Best Val Accuracy: {best_acc:.4f}")
