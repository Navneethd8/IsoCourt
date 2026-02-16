import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import os
import sys
from tqdm import tqdm

from dataset import FineBadmintonDataset
from model import CNN_LSTM_Model

def train_full(
    data_root, 
    list_file, 
    epochs=50, 
    batch_size=4, # End-to-end is memory heavy
    lr=1e-4,      # Lower LR for end-to-end
    device="cpu",
    hidden_size=128,
    save_path=None
):
    _dir = os.path.dirname(os.path.abspath(__file__))
    if save_path is None:
        save_path = os.path.join(_dir, "models", "badminton_model_full.pth")
    # Step 1: Augmentation Pipeline
    # Using v2 transforms which apply the same parameters to all frames in a list/tensor
    train_transform = v2.Compose([
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.RandomGrayscale(p=0.1),
        # Normalization is handled inside the model.forward for consistency
    ])

    # Step 2: DataLoader with WeightedRandomSampler for balance
    print(f"Loading dataset from {data_root}...")
    dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)
    
    # --- WeightedRandomSampler for Class Balance ---
    # We target stroke_type as the primary task for balancing
    st_labels = []
    print("Pre-calculating class weights for balanced sampling...")
    for sample in dataset.samples:
        labels = dataset._map_labels(sample)
        st_labels.append(labels['stroke_type'])
    # --- 80/20 Train/Val Split ---
    from torch.utils.data import Subset
    num_samples = len(dataset)
    indices = torch.randperm(num_samples).tolist()
    split = int(0.8 * num_samples)
    train_subset = Subset(dataset, indices[:split])
    val_subset = Subset(dataset, indices[split:])
    print(f"Split: {len(train_subset)} train / {len(val_subset)} val samples")
    
    # WeightedRandomSampler on train split only
    train_st_labels = torch.tensor([st_labels[i] for i in indices[:split]])
    class_counts = torch.bincount(train_st_labels)
    class_weights = 1. / (class_counts.float() + 1e-6)
    sample_weights = class_weights[train_st_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    num_workers = 0
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers, 
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Step 3: Model & Loss
    # MTL task classes
    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size, pretrained=True).to(device)
    
    # Partial Freeze: Freeze everything EXCEPT layer4 and the heads
    print("Freezing CNN backbone (Except Layer 4 for visual invariance)...")
    for name, param in model.cnn.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # Optimizer: Differential Learning Rates
    optimizer = optim.Adam([
        {'params': model.cnn.parameters(), 'lr': lr * 0.1},
        {'params': model.lstm.parameters(), 'lr': lr * 5},
        {'params': model.heads.parameters(), 'lr': lr * 5}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Class Balancing for stroke_type
    weights_st = torch.tensor([1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0]).to(device)
    criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
    criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    accumulation_steps = 4
    
    print(f"\nStarting End-to-End Training ({epochs} epochs)...")
    print(f"Differential LR: CNN={lr*0.1:.6f}, Heads={lr*5:.6f}")
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(frames)
            
            batch_loss = torch.tensor(0.0, device=device)
            loss_weights = {
                "stroke_type": 2.0, "position": 1.0, "technique": 0.5,
                "placement": 0.5, "intent": 0.5, "quality": 0.5
            }
            
            for task, logits in outputs.items():
                crit = criterion_st if task == "stroke_type" else criterion_default
                loss = crit(logits, labels[task])
                batch_loss += loss * loss_weights.get(task, 1.0)
            
            (batch_loss / accumulation_steps).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += batch_loss.item()
            pbar.set_postfix({'loss': running_loss/(batch_idx+1)})

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)
        
        # --- Validation Phase ---
        model.eval()
        val_correct = {k: 0 for k in task_classes.keys()}
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(frames)
                
                val_total += frames.size(0)
                for task, logits in outputs.items():
                    _, predicted = torch.max(logits.data, 1)
                    val_correct[task] += (predicted == labels[task]).sum().item()
        
        val_acc = 100 * val_correct["stroke_type"] / val_total
        pos_acc = 100 * val_correct["position"] / val_total
        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Val Type Acc: {val_acc:.1f}% | Val Pos Acc: {pos_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (Type Acc: {best_acc:.1f}%)")
            
            # Update Model Registry
            import json, datetime
            registry_path = os.path.join(os.path.dirname(save_path), "model_registry.json")
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                registry = {"models": {}, "active_model": None}
            
            model_name = os.path.basename(save_path)
            registry["models"][model_name] = {
                "accuracy": round(best_acc, 2),
                "epoch": epoch + 1,
                "hidden_size": hidden_size,
                "timestamp": datetime.datetime.now().isoformat(),
                "script": "train_full.py"
            }
            # Don't override active_model if train_fast already set it
            # train_full model is saved under a different name
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            
        # Periodic saves
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")

    print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    list_file = os.path.join(current_dir, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_full(
        data_root=data_root, 
        list_file=list_file, 
        epochs=30, # End-to-end is slower, start with 30
        device=device
    )
