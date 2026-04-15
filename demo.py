from __future__ import annotations

import os
import modal
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import CLIPImageProcessor

from checkpoint import load_checkpoint
from config import CFG
from data import load_artifacts_from_hub, _hf_token
from inference import beam_search_decode
from modal_setup import app, image, volume
from model import CLIPTransformerCaptioner
from vocabulary import Vocabulary

@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=60 * 30,
    volumes={"/model": volume},
)
def demo(image_id: Optional[str] = None, split_name: str = "test"):
    """
    Inference demo function
    
    Args:
        image_id: Specific image ID to caption (optional)
        split_name: Dataset split to use (train/validation/test)
        
    Returns:
        Dictionary with image_id and generated caption
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids, train_descriptions, val_descriptions, vocab = load_artifacts_from_hub(CFG.hf_repo_id)
    processor = CLIPImageProcessor.from_pretrained(CFG.clip_model_name)
    dataset = load_dataset(CFG.hf_repo_id, token=_hf_token())
    split = dataset[split_name]

    # Load model
    model = CLIPTransformerCaptioner(
        vocab_size=len(vocab),
        clip_model_name=CFG.clip_model_name,
        d_model=CFG.d_model,
        nhead=CFG.nhead,
        num_decoder_layers=CFG.num_decoder_layers,
        dim_feedforward=CFG.dim_feedforward,
        dropout=CFG.dropout,
        max_len=CFG.max_len,
        freeze_clip=False,
        use_lora=CFG.use_lora,
        lora_r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        lora_target_modules=CFG.lora_target_modules,
    ).to(device)

    ckpt, vocab = load_checkpoint(CFG.best_model_path, model, optimizer=None, map_location=device)
    model.eval()

    # Select image
    if image_id is not None:
        matches = [ex for ex in split if ex["image_id"] == image_id]
        if not matches:
            raise ValueError(f"image_id={image_id} not found in split={split_name}")
        sample = matches[0]
    else:
        sample = split[0]

    # Generate caption
    image = sample["image"].convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    caption = beam_search_decode(
        model,
        pixel_values,
        vocab,
        device,
        beam_size=CFG.beam_size,
        max_len=CFG.max_decode_len,
        length_penalty=CFG.length_penalty,
        no_repeat_ngram_size=CFG.no_repeat_ngram_size,
    )

    return {
        "image_id": sample["image_id"],
        "caption": caption,
    }
    
@app.local_entrypoint()
def main(image_id: str = None):
    result = demo.remote(image_id)
    print(result)