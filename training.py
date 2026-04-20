"""Training and evaluation loops"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CFG


def compute_loss(outputs, targets, criterion):
    """
    Compute cross-entropy loss
    
    Args:
        outputs: Model predictions [B, T, vocab_size]
        targets: Ground truth tokens [B, T]
        criterion: Loss function
        
    Returns:
        Scalar loss
    """
    return criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip: float = 1.0, scheduler=None):
    """
    Train for one epoch
    
    Args:
        model: Image captioner model
        loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cpu or cuda)
        scaler: AMP gradient scaler (optional)
        grad_clip: Gradient clipping value
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for i, (images, captions, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):   # 👈 dùng API mới
                outputs = model(images, captions_in)
                loss = compute_loss(outputs, captions_out, criterion)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
        else:
            outputs = model(images, captions_in)
            loss = compute_loss(outputs, captions_out, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)

        if (i + 1) % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.1e}"
            })

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation set
    
    Args:
        model: Image captioner model
        loader: Validation dataloader
        criterion: Loss function
        device: Device (cpu or cuda)
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for i, (images, captions, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        outputs = model(images, captions_in)
        loss = compute_loss(outputs, captions_out, criterion)

        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix(val_loss=f"{avg_loss:.4f}")

    return total_loss / len(loader)


def set_clip_phase(model, optimizer, epoch: int):
    """
    Manage CLIP freeze/unfreeze schedule and learning rates
    
    Args:
        model: Image captioner model
        optimizer: Optimizer
        epoch: Current epoch number
    """
    unfreeze = epoch >= CFG.freeze_clip_epochs
    model.set_clip_trainable(unfreeze)

    clip_lr = CFG.lr * CFG.clip_lr_scale if unfreeze else 0.0
    optimizer.param_groups[1]["lr"] = clip_lr

    state = "UNFROZEN" if unfreeze else "FROZEN"
    print(f"CLIP phase at epoch {epoch + 1}: {state} | clip_lr={clip_lr}")
