import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

import model

block_size = 200
batch_size = 32
eval_iter = 200
eval_interval = 300
n_emb = 120
n_head = 6
n_layer = 4
dropout = 0.2
max_iters = 4000
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher%27s%20Stone.txt"
output_file = "Book 1 - The Philosopher's Stone.txt"

response = requests.get(url)
response.raise_for_status()

with open(output_file, "wb") as file:
    file.write(response.content)

with open(output_file, 'r', encoding='utf-8') as f:
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