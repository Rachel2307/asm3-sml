# COMP SCI Assignment 3 – Q5: Small Language Models (RNN vs Transformer)
# ------------------------------------------------------------
# Step 1: Train a 10k BPE tokenizer (SentencePiece)
# Step 2: Train an LSTM and a Transformer on the SAME tokenized dataset
# Step 3: Report validation loss, perplexity, and runtime comparison
#
# ------------------------------------------------------------

import math, time, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# ==========  PART 1 — BPE Tokenizer =========
# ============================================
def train_bpe_tokenizer():
    import sentencepiece as spm
    INPUT = "input.txt"
    MODEL_PREFIX = "bpe10k"
    VOCAB_SIZE = 10_000

    if Path("bpe_tokens.pt").exists() and Path("bpe_vocab_size.txt").exists():
        print("Existing BPE model found, skipping retrain.")
        return

    assert Path(INPUT).exists(), f"Missing {INPUT}. Please provide a training text file."

    print(f"Training new BPE tokenizer (vocab={VOCAB_SIZE}) on {INPUT} ...")
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
    print(f"Trained BPE (vocab={sp.GetPieceSize()}) — saved bpe_tokens.pt ({len(ids)} tokens).")


# ============================================
# ==========  PART 2 — Data Setup ============
# ============================================
def load_data():
    assert Path("bpe_tokens.pt").exists(), "Run tokenizer training first."
    tokens = torch.load("bpe_tokens.pt")
    vocab_size = int(Path("bpe_vocab_size.txt").read_text())
    n = int(0.9 * len(tokens))
    return tokens[:n], tokens[n:], vocab_size


# ============================================
# ==========  PART 3 — Models ================
# ============================================
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, targets=None):
        x = self.embed(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, n_embd, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ============================================
# ==========  PART 4 — Training ==============
# ============================================
def get_batch(data, split, batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, device, num_batches=100):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        xb, yb = get_batch(data, "val", batch_size, block_size, device)
        _, loss = model(xb, yb)
        total += loss.item()
    model.train()
    val_loss = total / num_batches
    ppl = math.exp(val_loss)
    return val_loss, ppl


def train_model(model, optimizer, name, train_data, val_data, batch_size, block_size, device,
                train_steps=1000, eval_every=200, patience=5, min_delta=1e-3):
    best_val, best_ppl = float("inf"), float("inf")
    no_improve = 0
    start = time.time()

    for step in range(1, train_steps + 1):
        xb, yb = get_batch(train_data, "train", batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            val_loss, ppl = evaluate(model, val_data, batch_size, block_size, device)
            improved = (best_val - val_loss) > min_delta
            print(f"{name:11s} step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | ppl {ppl:.2f} {'↑improved' if improved else '—'}")
            if improved:
                best_val, best_ppl, no_improve = val_loss, ppl, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"{name}: early stopping (no val improvement for {patience} evals).")
                    break

    elapsed = time.time() - start
    return best_val, best_ppl, elapsed


# ============================================
# ==========  MAIN EXECUTION =================
# ============================================
if __name__ == "__main__":
    train_bpe_tokenizer()  # train BPE if not already done
    train_data, val_data, vocab_size = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); random.seed(1337)

    batch_size, block_size = 16, 64

    lstm = LSTMLanguageModel(vocab_size).to(device)
    transformer = TransformerModel(vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=block_size).to(device)

    opt_rnn = torch.optim.AdamW(lstm.parameters(), lr=3e-4)
    opt_trf = torch.optim.AdamW(transformer.parameters(), lr=3e-4)

    print("\nTraining LSTM...")
    lstm_val, lstm_ppl, lstm_time = train_model(lstm, opt_rnn, "LSTM", train_data, val_data, batch_size, block_size, device)

    print("\nTraining Transformer...")
    trf_val, trf_ppl, trf_time = train_model(transformer, opt_trf, "Transformer", train_data, val_data, batch_size, block_size, device)

    print("\n=== Comparison (Validation) ===")
    print(f"LSTM        : val_loss={lstm_val:.4f} | ppl={lstm_ppl:.2f} | time={lstm_time:.1f}s")
    print(f"Transformer : val_loss={trf_val:.4f} | ppl={trf_ppl:.2f} | time={trf_time:.1f}s")
