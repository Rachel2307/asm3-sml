# char_transformer_language_model.py
# Trains LSTM and a small Transformer on the SAME BPE(10k) tokenized dataset
# and reports validation loss + perplexity for both.

import math, time, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337); random.seed(1337)

# -------------------- Data --------------------
assert Path("bpe_tokens.pt").exists(), "Run: python tokenizer_bpe.py first."
tokens = torch.load("bpe_tokens.pt")  # 1D tensor of token ids
vocab_size = int(Path("bpe_vocab_size.txt").read_text())  # should be 10000

n = int(0.9 * len(tokens))
train_data, val_data = tokens[:n], tokens[n:]

batch_size = 16
block_size = 64

def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, num_batches: int = 100):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        xb, yb = get_batch("val")
        _, loss = model(xb, yb)
        total += loss.item()
    model.train()
    val_loss = total / num_batches
    ppl = math.exp(val_loss)
    return val_loss, ppl

# -------------------- Models --------------------
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
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(0.1),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.sa = MultiHeadAttention(n_head, n_embd, block_size)
        self.ln2 = nn.LayerNorm(n_embd); self.ff = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
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

# -------------------- Training --------------------
def train_model(
    model,
    optimizer,
    name,
    train_steps=10000,     # allow plenty, we'll stop early
    eval_every=200,
    patience=5,            # stop if no val improvement for 5 evals
    min_delta=1e-3         # require at least this improvement
):
    best_val = float("inf")
    best_ppl = float("inf")
    no_improve = 0
    t0 = time.time()

    for step in range(1, train_steps + 1):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            val_loss, ppl = evaluate(model, num_batches=50)
            improved = (best_val - val_loss) > min_delta
            status = "↑improved" if improved else "—"
            print(f"{name:11s} step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} | ppl {ppl:.2f} {status}")

            if improved:
                best_val, best_ppl = val_loss, ppl
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"{name}: early stopping (no val improvement for {patience} evals).")
                    break

    duration = time.time() - t0
    return best_val, best_ppl, duration


lstm = LSTMLanguageModel(vocab_size).to(device)
transformer = TransformerModel(vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=block_size).to(device)
opt_rnn = torch.optim.AdamW(lstm.parameters(), lr=3e-4)
opt_trf = torch.optim.AdamW(transformer.parameters(), lr=3e-4)

print("\nTraining LSTM...")
lstm_val, lstm_ppl, lstm_time = train_model(lstm, opt_rnn, "LSTM", train_steps=1000, eval_every=200)
print("\nTraining Transformer...")
trf_val, trf_ppl, trf_time = train_model(transformer, opt_trf, "Transformer", train_steps=1000, eval_every=200)

print("\n=== Comparison (Validation) ===")
print(f"LSTM        : val_loss={lstm_val:.4f} | ppl={lstm_ppl:.2f} | time={lstm_time:.1f}s")
print(f"Transformer : val_loss={trf_val:.4f} | ppl={trf_ppl:.2f} | time={trf_time:.1f}s")
