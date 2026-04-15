"""Model architecture: CLIP encoder + Transformer decoder"""

from typing import Tuple

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


def generate_square_subsequent_mask(sz: int, device):
    """Generate causal mask for Transformer decoder"""
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


class CLIPTransformerCaptioner(nn.Module):
    """
    Image captioner combining CLIP vision encoder with Transformer decoder
    Supports optional LoRA fine-tuning on CLIP
    """
    
    def __init__(
        self,
        vocab_size: int,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 30,
        freeze_clip: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    ):
        """
        Initialize CLIP + Transformer captioner
        
        Args:
            vocab_size: Size of vocabulary
            clip_model_name: Name of CLIP model
            d_model: Hidden dimension of decoder
            nhead: Number of attention heads
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            freeze_clip: Whether to freeze CLIP initially
            use_lora: Whether to use LoRA for CLIP fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA to
        """
        super().__init__()

        # Load CLIP vision model
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden = self.clip.config.hidden_size

        # Optional LoRA on CLIP vision transformer
        self.use_lora = use_lora
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(lora_target_modules),
                    bias="none",
                )
                self.clip = get_peft_model(self.clip, lora_config)
                self.clip.print_trainable_parameters()
            except Exception as e:
                print(f"[WARN] LoRA init failed, fallback to normal CLIP fine-tuning. Reason: {e}")
                self.use_lora = False

        # Projection and embedding layers
        self.visual_proj = nn.Linear(clip_hidden, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.max_len = max_len
        self.set_clip_trainable(not freeze_clip)

    def set_clip_trainable(self, trainable: bool):
        """Enable or disable CLIP parameter gradients"""
        for p in self.clip.parameters():
            p.requires_grad = trainable

    def forward(self, images, captions_in):
        """
        Forward pass
        
        Args:
            images: Batch of images [B, 3, H, W]
            captions_in: Input caption tokens [B, T]
            
        Returns:
            Logits [B, T, vocab_size]
        """
        device = images.device
        B, T = captions_in.shape

        # Get CLIP vision features
        clip_trainable = any(p.requires_grad for p in self.clip.parameters())
        if clip_trainable:
            vision = self.clip(pixel_values=images).last_hidden_state
        else:
            with torch.no_grad():
                vision = self.clip(pixel_values=images).last_hidden_state

        memory = self.visual_proj(vision)
        
        # Embed captions with positional embeddings
        positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)
        tgt = self.token_embedding(captions_in) + self.pos_embedding(positions)
        tgt = self.dropout(tgt)

        # Causal mask and padding mask
        tgt_mask = generate_square_subsequent_mask(T, device)
        tgt_key_padding_mask = captions_in.eq(0)

        # Decode
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc_out(decoded)
        return logits
