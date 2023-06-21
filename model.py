# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_emb, head_size, bias=False)
    self.query = nn.Linear(n_emb, head_size, bias=False)
    self.value = nn.Linear(n_emb, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)
    wei = q@k.transpose(-2, -1) # (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei) # (B, T, T)
    v = self.value(x) # (B, T, head_size)
    out = wei@v # (B, T, head_size)
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_emb, n_emb)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):

  def __init__(self, n_emb):
    super().__init__()
    self.nets = nn.Sequential([
        nn.Linear(n_emb, 4*n_emb),
        nn.ReLU(),
        nn.Linear(4*n_emb, n_emb),
        nn.Dropout(dropout),
    ])

  def forward(self, x):
    return self.nets(x)

class Block(nn.Module):

  def __init__(self, n_emb, n_head):
    super().__init__()
    head_size = n_emb//n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_emb)
    self.ln1 = nn.LayerNorm(n_emb)
    self.ln2 = nn.LayerNorm(n_emb)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class TinyGPT(nn.Module):

  def __init__(self, voc_size, n_emb):
    super().__init__()
    self.token_embedding_table = nn.Embedding(voc_size, n_emb)
    self.pos_embedding_table = nn.Embedding(block_size, n_emb)
    self.blocks = nn.Sequential([
        *[Block(n_emb, n_head) for _ in range(n_block)]
    ])
    self.ln_f = nn.LayerNorm(n_emb)
    self.lm_head = nn.Linear(n_emb, voc_size)

  def forward(self, idx, target=None):
    B, T, C = idx.shape
    token_embed = self.token_embedding_table(idx)
    pos_embed = self.pos_embedding_table(torch.arange(T, device=device))
    x = token_embed + pos_embed
    x = self.blocks(x)
    x = self.in_f(x)
    logits = self.lm_head(x)

    if target is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      target = target.view(B*T)
      loss = F.cross_entropy(logits, target)

    return logits, loss

  def generate(self, idx, max_tokens):
    for _ in range(max_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond) # logits (B, T, C)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, n_sample=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx