#!/usr/bin/env python
# coding: utf-8

# %%


# 'mkdir -p ../datasets && cd ../datasets && if [ ! -f "input.txt" ]; then curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt; fi && cd ../jupyter')


# %%


# read it in to inspect it
with open('../datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f'length of dataset in characters: {len(text):,}')


# %%



# %%


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


import torch
data = torch.tensor(encode(text), dtype=torch.long)


# %%


print(data.shape, data.dtype)
print(data[:1000])


# %%


# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# %%


block_size = 8
train_data[:block_size + 1]


# %%


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target  = y[t]
    print(f"when input is {context} the target: {target}")


# %%


torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


# %%


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# %%


xb, yb = get_batch('train')


# %%


print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")


# `self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`は、**各単語（トークン）に対する「次にどの単語が来るか」の予測スコア（ロジット）を格納するルックアップテーブル**を作成する役割を担っています。
# 
# このBigramモデルの文脈で、もう少し具体的に解説します。
# 
# ---
# 
# ### ## `nn.Embedding`の基本
# 
# `nn.Embedding`は、PyTorchにおいて単語のような離散的なIDを、密なベクトル表現（埋め込みベクトル）に変換するためのレイヤーです。基本的には巨大な行列であり、**単語のIDをインデックスとして、対応する行ベクトルを抜き出す**という動作をします。
# 
# `nn.Embedding(num_embeddings, embedding_dim)` の引数は以下の通りです。
# * `num_embeddings`: 埋め込む単語の総数。通常は語彙サイズ (`vocab_size`) を指定します。
# * `embedding_dim`: 各単語を表現するベクトルの次元数。
# 
# ---
# 
# ### ## このモデルにおける`nn.Embedding`の特別な役割
# 
# この`BigramLanguageModel`では、`embedding_dim`に`vocab_size`が指定されています。これが重要なポイントです。
# 
# `self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`
# 
# これは、サイズが `(vocab_size, vocab_size)` の行列を作成することを意味します。
# 
# * **行 (Rows)**: `vocab_size`個の行は、語彙に含まれる各トークンに対応します。例えば、0番目の行はIDが0のトークン、1番目の行はIDが1のトークン...という具合です。
# * **列 (Columns)**: `vocab_size`個の列を持つベクトル（各行）が、**次にどのトークンが来るかの予測スコア（ロジット）**を表します。
# 
# #### ### 処理の流れ
# 
# 1.  **入力**: `forward`メソッドに、トークンIDのシーケンス `idx` (shape: `(B, T)`、Bはバッチサイズ, Tはシーケンス長) が入力されます。
# 2.  **ルックアップ**: `self.token_embedding_table(idx)` が実行されると、`idx` の各トークンIDに対応する行ベクトルがテーブルから抜き出されます。
# 3.  **出力 (`logits`)**: 出力される `logits` のshapeは `(B, T, vocab_size)` となります。これは、入力シーケンスの各トークンに対して、「次に来る単語の候補（全`vocab_size`個）」それぞれの予測スコアを持っていることを意味します。
# 
# 
# 
# #### ### なぜこれがBigramモデルなのか？
# 
# **Bigramモデル**は、**直前の1つの単語だけを見て次の単語を予測**します。
# このコードはまさにその考え方を実装しています。
# 
# * あるトークン（例: "hello"）が入力されると、`nn.Embedding`テーブルから "hello" に対応する行を引いてきます。
# * その行ベクトル（サイズ `vocab_size`）が、次に "world" が来る確率、"there" が来る確率、"!" が来る確率...といった、語彙の全単語に対する予測スコア（ロジット）そのものになるわけです。
# 
# つまり、この `nn.Embedding` レイヤーは、単に単語をベクトル化するだけでなく、**「ある単語」から「次の単語の予測ロジット」へのマッピング**という、モデルの予測機能そのものを担っています。これがこのモデルの核心部分です。👍

# %%


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

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



m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_token=100)[0].tolist()))


# %%


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


# %%


batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb, = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())


# %%


print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_token=500)[0].tolist()))


# # The mathematical trick in self-attention

# %%


torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape


# %%


# We want x[b, t] = mean_{i<=t} x[b, i]
# version 1
xbow = torch.zeros((B, T, C)) # x bag_of_words
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)


# %%


# version 2
wei = torch.tril(torch.ones((T,T)))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x     # (B, T, T) @ (B, T, C) ---> (B, T, C)

# check xbow == xbow2
torch.max(xbow - xbow2)


# %%


# version3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.max(xbow - xbow3)


# In[ ]:


torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b

print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)


# In[ ]:




