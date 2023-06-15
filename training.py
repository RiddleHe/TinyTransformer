# -*- coding: utf-8 -*-
"""training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gLIEfwR6G1pc46QrtHkPGbqyGa0BJTCJ
"""

# Commented out IPython magic to ensure Python compatibility.
# %run model.ipynb

block_size = 256
batch_size = 64
eval_iter = 200
eval_interval = 300
n_emb = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 5000
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset

with open('wittgenstein.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
voc_size = len(chars)
print("".join(chars))

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda c: "".join([itos[ix] for ix in c])

print(encode("Hi there"))
print(decode(encode("Hi there")))

data = torch.tensor(encode(text), dtype=torch.long,)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split=='train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  X = torch.stack([data[i:i+block_size] for i in ix],)
  Y = torch.stack([data[i+1:i+block_size+1] for i in ix],)
  X, Y = X.to(device), Y.to(device)
  return X,Y

xb, yb = get_batch('train')
print(xb)

@torch.no_grad()
def eval_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iter)
    for k in range(eval_iter):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss
    out[split] = losses.mean()
  model.train()
  return out

model = TinyGPT()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for i in range(max_iters):

  if i%eval_interval==0:
    losses = eval_loss()
    print(f"Step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())

input = torch.zeros((1,1), dtype=torch.long, device=device)
out = m.generate(idx=input, max_tokens=5000)[0].tolist()
print(decode(out))