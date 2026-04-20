"""Checkpoint management for training"""

import os
from dataclasses import asdict

import torch

from config import Config, CFG
from utils import ensure_dir
from vocabulary import Vocabulary
from huggingface_hub import upload_file


def save_checkpoint(path: str, model, optimizer, epoch: int, loss: float, vocab: Vocabulary, cfg: Config):
    """
    Save training checkpoint
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state (can be None)
        epoch: Current epoch
        loss: Current loss
        vocab: Vocabulary instance
        cfg: Configuration
    """
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
            "vocab_state": vocab.to_dict(),
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    """
    Load training checkpoint
    
    Args:
        path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        map_location: Device location for loading
        
    Returns:
        Tuple of (checkpoint_dict, vocab)
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=False)
    
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[WARN] Could not load optimizer state: {e}")
    
    vocab = Vocabulary.from_dict(ckpt["vocab_state"])
    return ckpt, vocab


def try_resume_training(checkpoint_path, model, optimizer, device):
    """
    Try to resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to resume
        optimizer: Optimizer to resume
        device: Device (cpu or cuda)
        
    Returns:
        Tuple of (start_epoch, best_loss, vocab)
    """
    if os.path.exists(checkpoint_path):
        print(f"🔁 Found checkpoint: {checkpoint_path} → Resuming...")
        ckpt, vocab = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            map_location=device,
        )
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        print(f"   Resumed from epoch {ckpt['epoch']} | loss={best_loss:.6f}")
        return start_epoch, best_loss, vocab
    
    print("🆕 No checkpoint found → Training from scratch")
    return 0, float("inf"), None

def upload_best_model():
    repo_id = "hellloooworlddd123/imageDescription"  

    file_path = CFG.best_model_path

    if not os.path.exists(file_path):
        print("Best model not found!!!")
        return

    upload_file(
        path_or_fileobj=file_path,
        path_in_repo="models/clip_transformer_best.pth",  # 👈 đặt folder đẹp
        repo_id=repo_id,
        repo_type="dataset",  # hoặc "model"
        token=os.getenv("HF_TOKEN"),
    )

    print(" Uploaded best model to HuggingFace!")
