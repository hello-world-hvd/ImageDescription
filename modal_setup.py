"""Modal container and volume setup"""

import os
import modal

# Initialize Modal app
app = modal.App("clip_train")

# Define image with dependencies and source code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "huggingface_hub",
        "pillow",
        "numpy",
        "tqdm",
        "matplotlib",
        "peft",
        "accelerate",
    )
    .add_local_dir(".", remote_path="/root", ignore=[".venv", "__pycache__", ".git"])
)

# Create or reference volume for model checkpoints
volume = modal.Volume.from_name("clip-caption-checkpoints_v2", create_if_missing=True)
