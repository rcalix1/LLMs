# ## A small ChatGPT style Transformer
# * concepts in NLP and Transformers 
# * generative mdoels

############################################################ 

import torch
import numpy as np
import requests
## import tiktoken
import torch.nn as nn

from torch.nn import functional as F

############################################################

## !pip install requests
## !pip install tiktoken    ## requires python   >    3.9

############################################################

torch.manual_seed(1337)

block_size = 8      ## max content length for predictions
batch_size = 32 
max_iters  = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

############################################################

input_file_path = 'input.txt'

## data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
############################################################

print("length of data in characters")
len(text)

############################################################

chars = sorted(     list(set(text))   )
vocab_size = len(chars)

print(  ''.join(chars)  )

############################################################# 
## tokenizer

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [ stoi[c]          for c in s   ]    ## encoder: string to integer
decode = lambda l: ''.join(   itos[i] for i in l   )    ## decoder: interger to string

#############################################################

data = torch.tensor(   encode(text), dtype=torch.long   )
n    = int(   0.9*len(data)   )
train_data = data[:n]
val_data   = data[n:]

#############################################################

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    ix = torch.randint(   len(data) - block_size, (batch_size,)   )
    x  = torch.stack(    [  data[ i : i+block_size ]   for i in ix]    ) 
    y  = torch.stack(    [  data[ i+1 : i+1+block_size ]   for i in ix]    )
    
    x, y = x.to(device), y.to(device)

    return x, y

############################################################

@torch.no_grad()    ## for efficiency
def estimate_loss():
    out = {}
    model.eval()   ## no training
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  ## back to training
    return out

############################################################


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx)      ## batch, time, vocab_size (4, 8, 65)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets  = targets.view(B*T)
            loss   = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        ## idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            logits, loss = self(idx)
            ## focus only on last time stamp
            logits = logits[:, -1, :]           ## becomes (B, C)
            probs = F.softmax(logits, dim= -1)    ## (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)     ## (B, 1)
            ## append sample to the running sequence
            idx = torch.cat(  (idx, idx_next), dim=1  )            ## (B, T+1)
        return idx
            
            
            
######################################################################


model   = BigramLanguageModel(vocab_size)
m = model.to(device)

######################################################################

optimizer = torch.optim.Adam(  m.parameters(), lr=learning_rate   )

######################################################################

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    
    ## evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)   ## zero out
    loss.backward()
    optimizer.step()
    

################################################################
#### now, regenerate after some training


## Kick off generation with some starting token. In this case id 0

context = torch.zeros(  (1, 1),  dtype=torch.long, device=device   )

gen_text = m.generate(context, max_new_tokens=500)[0].tolist()

print(  decode(gen_text)   )









