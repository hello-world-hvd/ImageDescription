"""Data loading and dataset classes"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from config import CFG
from vocabulary import Vocabulary
from utils import load_json, load_pickle


def _hf_token() -> Optional[str]:
    """Get Hugging Face token from environment"""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _hf_download(repo_id: str, filename: str) -> str:
    """Download file from Hugging Face Hub"""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=_hf_token(),
    )


def load_artifacts_from_hub(repo_id: str):
    """
    Load training artifacts from Hugging Face Hub
    
    Args:
        repo_id: Hugging Face dataset repository ID
        
    Returns:
        Tuple of (train_ids, val_ids, train_descriptions, val_descriptions, vocab)
    """
    train_ids = load_pickle(_hf_download(repo_id, CFG.train_ids_filename))
    val_ids = load_pickle(_hf_download(repo_id, CFG.val_ids_filename))
    train_descriptions = load_pickle(_hf_download(repo_id, CFG.train_desc_filename))
    val_descriptions = load_pickle(_hf_download(repo_id, CFG.val_desc_filename))

    vocab_path = _hf_download(repo_id, CFG.vocab_filename)
    vocab_state = load_json(vocab_path)
    vocab = Vocabulary.from_dict(vocab_state)
    
    return train_ids, val_ids, train_descriptions, val_descriptions, vocab


class CaptionDataset(Dataset):
    """Dataset for image captioning"""
    
    def __init__(
        self,
        hf_split,
        image_names: List[str],
        captions: List[str],
        vocab: Vocabulary,
        processor,
        max_len: int = 30,
    ):
        """
        Initialize caption dataset
        
        Args:
            hf_split: Hugging Face dataset split
            image_names: List of image IDs
            captions: List of caption texts
            vocab: Vocabulary instance
            processor: CLIP image processor
            max_len: Maximum caption length
        """
        self.hf_split = hf_split
        self.image_names = image_names
        self.captions = captions
        self.vocab = vocab
        self.processor = processor
        self.max_len = max_len
        self.image_index = {ex["image_id"]: i for i, ex in enumerate(hf_split)}

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.captions)

    def __getitem__(self, idx: int):
        """Get single sample (image, caption)"""
        image_name = self.image_names[idx]
        caption = self.captions[idx]

        # Get image from HF dataset
        row = self.hf_split[self.image_index[image_name]]
        image = row["image"].convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Numericalize caption with BOS/EOS tokens
        cap_ids = [self.vocab.word2idx["<bos>"]]
        cap_ids.extend(self.vocab.numericalize(caption))
        cap_ids.append(self.vocab.word2idx["<eos>"])

        # Pad or truncate to max_len
        if len(cap_ids) < self.max_len:
            cap_ids.extend([self.vocab.word2idx["<pad>"]] * (self.max_len - len(cap_ids)))
        else:
            cap_ids = cap_ids[: self.max_len]
            cap_ids[-1] = self.vocab.word2idx["<eos>"]

        caption_tensor = torch.tensor(cap_ids, dtype=torch.long)
        return pixel_values, caption_tensor, image_name


def collate_batch(batch, pad_idx: int):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from dataset
        pad_idx: Padding token index
        
    Returns:
        Tuple of (images, captions, image_names)
    """
    images, captions, names = zip(*batch)
    images = torch.stack(images)
    captions = torch.stack(captions)
    return images, captions, list(names)
