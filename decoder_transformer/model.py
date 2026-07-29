import config
import dataset
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(config.context_size, config.context_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        out = x + self.ffwd(self.ln2(x))

        return out


class DecoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(dataset.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.context_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_heads) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, dataset.vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd) batches, times, embedding dim
        pos_emb = self.position_embedding_table(torch.arange(T, device = config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, dataset.vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.context_size:] #only last -context_size- tokens from every sequence
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
