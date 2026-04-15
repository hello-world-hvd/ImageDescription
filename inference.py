"""Inference functions including beam search decoding"""

from typing import List

import torch

from model import generate_square_subsequent_mask
from vocabulary import Vocabulary


@torch.no_grad()
def beam_search_decode(
    model,
    image_tensor,
    vocab: Vocabulary,
    device,
    beam_size: int = 3,
    max_len: int = 30,
    length_penalty: float = 0.7,
    no_repeat_ngram_size: int = 2,
):
    """
    Beam search decoding for image captioning
    
    Args:
        model: CLIP Transformer captioner model
        image_tensor: Input image tensor [3, H, W]
        vocab: Vocabulary instance
        device: Device (cpu or cuda)
        beam_size: Beam search width
        max_len: Maximum caption length
        length_penalty: Length normalization penalty
        no_repeat_ngram_size: N-gram repetition penalty
        
    Returns:
        Decoded caption string
    """
    model.eval()

    bos = vocab.word2idx["<bos>"]
    eos = vocab.word2idx["<eos>"]
    pad = vocab.word2idx["<pad>"]

    # Prepare image
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Get CLIP vision features
    clip_trainable = any(p.requires_grad for p in model.clip.parameters())
    if clip_trainable:
        vision = model.clip(pixel_values=image_tensor).last_hidden_state
    else:
        with torch.no_grad():
            vision = model.clip(pixel_values=image_tensor).last_hidden_state

    memory = model.visual_proj(vision)

    # Initialize beams: (sequence, score)
    beams = [([bos], 0.0)]
    finished = []

    def has_repeat_ngram(seq: List[int], n: int) -> bool:
        if n <= 0 or len(seq) < 2 * n:
            return False
        ngrams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        return len(ngrams) != len(set(ngrams))

    # Beam search iterations
    for _ in range(max_len - 1):
        candidates = []
        
        for seq, score in beams:
            # Skip finished sequences
            if seq[-1] == eos:
                finished.append((seq, score))
                continue

            # Get next token probabilities
            tgt = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            positions = torch.arange(0, tgt.shape[1], device=device).unsqueeze(0)
            tgt_emb = model.token_embedding(tgt) + model.pos_embedding(positions)
            tgt_emb = model.dropout(tgt_emb)
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1], device)
            tgt_key_padding_mask = tgt.eq(pad)

            decoded = model.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            logits = model.fc_out(decoded[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            # Get top-k tokens
            topk = torch.topk(log_probs, beam_size)
            for token_id, token_logp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_seq = seq + [token_id]
                
                # Skip sequences with repeated n-grams
                if no_repeat_ngram_size > 0 and has_repeat_ngram(new_seq, no_repeat_ngram_size):
                    continue
                
                candidates.append((new_seq, score + token_logp))

        if not candidates:
            break

        # Select top-k candidates (normalized by length)
        def norm_score(item):
            seq, score = item
            lp = ((5 + len(seq)) / 6) ** length_penalty if length_penalty > 0 else 1.0
            return score / lp

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_size]

    # Finalize finished sequences
    finished.extend([b for b in beams if b[0][-1] == eos])
    if not finished:
        finished = beams

    finished.sort(key=lambda x: x[1] / (((5 + len(x[0])) / 6) ** length_penalty), reverse=True)
    best_seq = finished[0][0]

    # Convert token IDs to words
    words = []
    for token_id in best_seq:
        token = vocab.idx2word.get(token_id, "<unk>")
        if token in ("<bos>", "<pad>"):
            continue
        if token == "<eos>":
            break
        words.append(token)
    
    return " ".join(words)
