# Image Description with CLIP + Transformer

Image captioning with CLIP + Transformer, trained on Modal, data loaded from Hugging Face, and caption generation via beam search

## 📂 Project Structure

```
config.py           # Configuration settings (learning rate, model hyperparameters, etc.)
utils.py            # Utility functions (file I/O, seeding, data preparation)
vocabulary.py       # Vocabulary class for text tokenization
data.py             # Dataset and dataloader utilities
model.py            # CLIP-based image encoder + Transformer decoder architecture
inference.py        # Beam search decoding for caption generation
training.py         # Training and evaluation loops
checkpoint.py       # Checkpoint saving/loading functionality
modal_setup.py      # Modal container and volume configuration
train.py            # Main training
demo.py             # Reference
```

## 🚀 Quick Start

### Setup

1. Install dependencies (see `modal_setup.py` for full list):

   ```bash
   pip install torch transformers datasets huggingface_hub peft accelerate
   ```

2. Set up Hugging Face credentials:

   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. Prepare data on Hugging Face Hub with:
   - `data/train-*.parquet`, `data/validation-*.parquet`, `data/test-*.parquet`
   - `train_ids.pkl`, `val_ids.pkl`
   - `train_descriptions_proc.pkl`, `val_descriptions_proc.pkl`
   - `vocab.json`

### Training

For local testing:

```python
from train import train, demo
from utils import set_seed
from config import CFG

set_seed(CFG.seed)
train()  # Will run on Modal GPU if configured
```

For Modal:

```bash
modal run train.py
```

### Inference

```python
from train import demo

result = demo(image_id="image_123", split_name="test")
print(result)  # {"image_id": "image_123", "caption": "..."}
```

## 📋 Configuration

Edit `config.py` to adjust:

- **Data**: `hf_repo_id`, `freq_threshold`, `max_len`
- **Training**: `batch_size`, `num_epochs`, `lr`, `weight_decay`
- **Model**: `d_model`, `nhead`, `num_decoder_layers`, `dropout`
- **Inference**: `beam_size`, `max_decode_len`, `length_penalty`

## 🔧 Key Components

### Model Architecture

- **Encoder**: CLIP Vision Transformer (frozen initially, unfrozen after 3 epochs)
- **Optional LoRA**: Efficient fine-tuning with low-rank adaptation
- **Decoder**: Transformer decoder with multi-head attention + causal masking

### Training Features

- Teacher forcing with caption shifting
- Mixed precision (AMP) training on CUDA
- Learning rate scheduling with validation-based reduction
- Checkpoint saving (latest + best models)
- Gradient clipping and warm-up

### Inference

- Beam search with:
  - Length normalization penalty
  - N-gram repetition prevention
  - Efficient batching

## 📊 Data Format

### Vocabulary

- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Built from descriptions with configurable frequency threshold

### Dataset

- Input: Image (3×224×224 CLIP format) + caption tokens
- Output: Image embeddings + numericalized captions
- Padding/truncation to `max_len=30` tokens

## 🎯 Model Improvements

- **Gradual Unfreezing**: Keeps CLIP frozen initially for stability
- **Beam Search**: Generates more diverse and coherent captions
- **Modular Design**: Easy to extend and customize components

## 📊 Results

### Training Performance

The model was trained for **8 epochs** with a gradual unfreezing strategy for CLIP.

| Epoch | Train Loss | Val Loss        | CLIP     |
| ----- | ---------- | --------------- | -------- |
| 1     | 3.81       | 3.47            | Frozen   |
| 2     | 3.28       | 3.30            | Frozen   |
| 3     | 3.11       | 3.23            | Frozen   |
| 4     | 2.97       | ~3.20           | Frozen   |
| 5     | 2.87       | ~3.18           | Unfrozen |
| 6     | 2.79       | **3.17 (Best)** | Unfrozen |
| 7     | 2.70       | 3.19            | Unfrozen |
| 8     | 2.60       | 3.20            | Unfrozen |

### Output

{
'image_id': '000000244750.jpg',
'caption': 'a group of people sitting at a table with wine glasses'
}
