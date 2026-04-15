"""Main training and inference entry point with Modal integration"""

from __future__ import annotations

import os
import modal
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import CLIPImageProcessor

from checkpoint import load_checkpoint, save_checkpoint, try_resume_training
from config import CFG
from data import CaptionDataset, collate_batch, load_artifacts_from_hub, _hf_token
from inference import beam_search_decode
from modal_setup import app, image, volume
from model import CLIPTransformerCaptioner
from training import evaluate, set_clip_phase, train_one_epoch
from utils import build_pairs, ensure_dir, save_json, set_seed
from vocabulary import Vocabulary

def build_vocab_from_descriptions(train_descriptions: dict, freq_threshold: int) -> Vocabulary:
    """
    Build vocabulary from training descriptions
    
    Args:
        train_descriptions: Dictionary of descriptions
        freq_threshold: Minimum word frequency
        
    Returns:
        Vocabulary instance
    """
    train_caption_texts = [cap for caps in train_descriptions.values() for cap in caps]
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocab(train_caption_texts)
    return vocab




@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=60 * 60 * 24,
    volumes={"/model": volume},
)
def train():
    """
    Main training function to be executed on Modal
    """
    set_seed(CFG.seed)
    ensure_dir(CFG.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load HF artifacts
    train_ids, val_ids, train_descriptions, val_descriptions, vocab = load_artifacts_from_hub(CFG.hf_repo_id)
    print("train_descriptions: ", train_descriptions['COCO_train2014_000000000009.jpg'])
    # Fallback build vocab if hub vocab is missing / empty
    if len(vocab) <= 4:
        print("Vocab looks empty. Rebuilding from train descriptions...")
        vocab = build_vocab_from_descriptions(train_descriptions, CFG.freq_threshold)
        save_json(os.path.join(CFG.checkpoint_dir, CFG.vocab_cache_path.split("/")[-1]), vocab.to_dict())

    processor = CLIPImageProcessor.from_pretrained(CFG.clip_model_name)

    dataset = load_dataset(CFG.hf_repo_id, token=_hf_token())
    train_split = dataset["train"]
    val_split = dataset["validation"]

    train_image_names, train_captions = build_pairs(train_ids, train_descriptions)
    val_image_names, val_captions = build_pairs(val_ids, val_descriptions)

    train_dataset = CaptionDataset(
        hf_split=train_split,
        image_names=train_image_names,
        captions=train_captions,
        vocab=vocab,
        processor=processor,
        max_len=CFG.max_len,
    )
    val_dataset = CaptionDataset(
        hf_split=val_split,
        image_names=val_image_names,
        captions=val_captions,
        vocab=vocab,
        processor=processor,
        max_len=CFG.max_len,
    )

    pad_idx = vocab.word2idx["<pad>"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_idx),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate_batch(batch, pad_idx),
    )

    # Initialize model
    model = CLIPTransformerCaptioner(
        vocab_size=len(vocab),
        clip_model_name=CFG.clip_model_name,
        d_model=CFG.d_model,
        nhead=CFG.nhead,
        num_decoder_layers=CFG.num_decoder_layers,
        dim_feedforward=CFG.dim_feedforward,
        dropout=CFG.dropout,
        max_len=CFG.max_len,
        freeze_clip=True,
        use_lora=CFG.use_lora,
        lora_r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        lora_target_modules=CFG.lora_target_modules,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=CFG.label_smoothing,
    )

    # Two param groups: text-side + projection always trainable, CLIP lr initially 0.
    clip_params = list(model.clip.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("clip.")]

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": CFG.lr, "weight_decay": CFG.weight_decay},
            {"params": clip_params, "lr": 0.0, "weight_decay": CFG.weight_decay},
        ]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.use_amp and device.type == "cuda"))

    start_epoch, best_loss, loaded_vocab = try_resume_training(
        CFG.checkpoint_path,
        model,
        optimizer,
        device,
    )

    if loaded_vocab is not None:
        vocab = loaded_vocab

    # Training loop
    for epoch in range(start_epoch, CFG.num_epochs):
        set_clip_phase(model, optimizer, epoch)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, CFG.grad_clip
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"\n====== Epoch {epoch + 1:02d}/{CFG.num_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}======"
        )

        # Save latest checkpoint
        save_checkpoint(CFG.checkpoint_path, model, optimizer, epoch, val_loss, vocab, CFG)

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(CFG.best_model_path, model, optimizer, epoch, val_loss, vocab, CFG)
            print(f"  -> best model saved: {CFG.best_model_path}")
        
        volume.commit()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    volume.commit()
    print("Training completed.")


@app.local_entrypoint()
def main():
    """Local entry point to start training on Modal"""
    train.remote()


if __name__ == "__main__":
    main()
