from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    """Configuration for CLIP Transformer Image Captioner with LoRA"""
    
    seed: int = 42

    # Hugging Face dataset repo with:
    # - data/train-*.parquet
    # - data/validation-*.parquet
    # - data/test-*.parquet
    # - train_ids.pkl, val_ids.pkl, train_descriptions_proc.pkl, val_descriptions_proc.pkl, vocab.json
    hf_repo_id: str = "hellloooworlddd123/imageDescription"

    # Filenames stored at repo root
    train_ids_filename: str = "train_ids.pkl"
    val_ids_filename: str = "val_ids.pkl"
    train_desc_filename: str = "train_description_proc.pkl"
    val_desc_filename: str = "val_description_proc.pkl"
    vocab_filename: str = "vocab.json"

    # Data / caption settings
    freq_threshold: int = 5
    max_len: int = 30
    subset_size: Optional[int] = None

    # Dataloader
    batch_size: int = 32
    num_workers: int = 2

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    num_epochs: int = 12
    num_warmup_steps: int = 1200
    grad_clip: float = 1.0
    use_amp: bool = True

    # Freeze / unfreeze schedule
    freeze_clip_epochs: int = 3
    clip_lr_scale: float = 0.1
    clip_grad_clip: float = 1.0

    # Beam search / inference
    beam_size: int = 3
    max_decode_len: int = 30
    length_penalty: float = 0.7
    no_repeat_ngram_size: int = 2

    # Model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.2

    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")

    # Checkpoints on Modal Volume
    checkpoint_dir: str = "/model"
    checkpoint_path: str = "/model/clip_transformer_last.pth"
    best_model_path: str = "/model/clip_transformer_best.pth"
    vocab_cache_path: str = "/model/vocab.json"

CFG = Config()
