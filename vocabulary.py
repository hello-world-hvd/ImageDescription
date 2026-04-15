"""Vocabulary class for tokenization and numericalization"""

from collections import Counter
from typing import Dict, List

class Vocabulary:
    
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.idx = 0

        # Add special tokens
        for token in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            self.add_word(token)

    def __len__(self) -> int:
        return len(self.word2idx)

    def add_word(self, word: str) -> None:
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, sentence_list: List[str]) -> None:
        frequencies = Counter()
        for sentence in sentence_list:
            frequencies.update(sentence.split())
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.add_word(word)

    def numericalize(self, text: str) -> List[int]:
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in text.split()]

    def to_dict(self) -> Dict:
        return {
            "freq_threshold": self.freq_threshold,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "idx": self.idx,
        }

    @classmethod
    def from_dict(cls, state: Dict) -> "Vocabulary":
        vocab = cls(freq_threshold=int(state.get("freq_threshold", 5)))
        vocab.word2idx = {k: int(v) for k, v in state["word2idx"].items()}
        vocab.idx2word = {int(k): v for k, v in state["idx2word"].items()}
        vocab.idx = int(state["idx"])
        return vocab
