# tokenizer_bpe.py
# Trains a 10k BPE tokenizer on input.txt and saves tokens + vocab size.
# pip install sentencepiece

import sentencepiece as spm
import torch
from pathlib import Path

INPUT = "input.txt"
MODEL_PREFIX = "bpe10k"
VOCAB_SIZE = 10_000

def main():
    assert Path(INPUT).exists(), f"Missing {INPUT}"
    spm.SentencePieceTrainer.Train(
        input=INPUT,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{MODEL_PREFIX}.model")
    text = Path(INPUT).read_text(encoding="utf-8")
    ids = sp.EncodeAsIds(text)
    torch.save(torch.tensor(ids, dtype=torch.long), "bpe_tokens.pt")
    Path("bpe_vocab_size.txt").write_text(str(sp.GetPieceSize()))
    print(f"Trained BPE (vocab={sp.GetPieceSize()}). Saved bpe_tokens.pt ({len(ids)} ids).")

if __name__ == "__main__":
    main()
