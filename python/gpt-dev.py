# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

# %%
# 'mkdir -p ../datasets && cd ../datasets && if [ ! -f "input.txt" ]; then curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt; fi && cd ../jupyter')

# %%
# data loading
def get_batch(split:str) -> tuple[torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# %%
@torch.no_grad()
def estimate_loss(model:nn.Module) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %%
# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size:int) -> None:
        super().__init__()
        # each token directory reads of the logits for the net token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets:torch.Tensor|None=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)
        if target is None:
            loss = None
        elif targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_token):
        # idx is (B, T) array of indicies in the current context
        for _ in range(max_new_token):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # become (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# %%
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
train_val_split = 0.9

# %%
# reproducibility
torch.manual_seed(1337)

# %%
# read it in to inspect it
with open('../datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# %%
print(f'length of dataset in characters: {len(text):,}')
print(text[:1000]) # check first 1000 chars

# %%
# here are all the unique characters that occur in this text file
chars = sorted(list(set(text)))
vocab_size = len(chars)

# %%
print(''.join(chars))
print(vocab_size)

# %%
# create a mapping from charactors to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# %%
print(encode("hi! there"))
print(decode(encode("hi! there")))

# %%
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# %%
n = int(train_val_split * len(data))
train_data = data[:n]
val_data = data[n:]

# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target  = y[t]
    print(f"when input is {context} the target: {target}")
    
# %%
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# %%
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# %%
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb ,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
encoded_batch = m.generate(idx = context, max_new_token=500)
print(decode(encoded_batch[0].tolist()))
# %%
