"""Utility functions for data processing and file handling"""

import json
import os
import pickle
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility across numpy, torch, and random"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    """Save object to JSON file"""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str) -> Any:
    """Load pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_pairs(image_ids: List[str], descriptions: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    image_names: List[str] = []
    captions: List[str] = []
    for image_id in image_ids:
        for cap in descriptions[image_id]:
            image_names.append(image_id)
            captions.append(cap)
    return image_names, captions
