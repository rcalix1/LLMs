#!/usr/bin/env python
# coding: utf-8

# ## A small ChatGPT style Transformer
# 
# * concepts in NLP and Transformers
# 
# * generative mdoels
# 

# In[1]:


import torch
import numpy as np
import requests
## import tiktoken
import torch.nn as nn

from torch.nn import functional as F



# In[2]:


## !pip install requests
## !pip install tiktoken    ## requires python   >    3.9


# In[3]:


torch.manual_seed(1337)


# In[4]:


block_size = 8   ## max content length for predictions
batch_size = 4 


# In[5]:


input_file_path = 'input.txt'


data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
        


# In[6]:


with open(input_file_path, 'r') as f:
    text = f.read()
    



# In[7]:


print("length of data in characters")
len(text)


# In[8]:


## n = len(data)


# In[9]:


print(  text[:1000]   )


# 
# ## get list of unique characters
# 

# In[10]:


chars = sorted(     list(set(text))   )
chars


# In[11]:


vocab_size = len(chars)
vocab_size 


# In[12]:


print(  ''.join(chars)  )


# ## tokenizer
# 
# create a mapping from characters to integers
# 
# * other better options are tiptoken and SentencePiece
# 

# In[13]:


stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [ stoi[c]          for c in s   ]    ## encoder: string to integer
decode = lambda l: ''.join(   itos[i] for i in l   )    ## decoder: interger to string


# ## Encode and decode

# In[14]:


print(   encode("hii there")    )


# In[15]:


print(   decode(   encode("hii there")  )   )


# 
# ## Encode the text
# 

# In[16]:


data = torch.tensor(   encode(text), dtype=torch.long   )
print(data.shape, data.type)


# In[17]:


data


# In[18]:


print(   data[:1000]   )


# 
# ## Train and test split
# 

# In[19]:


n = int(   0.9*len(data)   )
n


# In[20]:


train_data = data[:n]
val_data   = data[n:]


# 
# ## sample random chunks of block size
# 

# In[21]:


train_data[:block_size+1]


# 
# ## offset by one
# 

# In[22]:


x = train_data[:block_size]
y = train_data[1:block_size+1]


# In[23]:


x


# In[24]:


y


# In[25]:


for t in range(block_size):
    context = x[: t+1]
    target  = y[t]
    print(f"when input is {context} then the target is: {target}")


# 
# ## batch processing
# 

# In[26]:


len(data)


# In[27]:


block_size


# In[28]:


len(data) - block_size


# In[29]:


(batch_size,) 


# In[30]:


ix = torch.randint(   len(data) - block_size, (batch_size,)   )
ix


# 
# ## Randomly sample batch size (e.g. 4) ids and get the next block_size ids (e.g. 7) for each respectively
# 

# In[31]:


def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    ix = torch.randint(   len(data) - block_size, (batch_size,)   )
    x  = torch.stack(    [  data[ i : i+block_size ]   for i in ix]    ) 
    y  = torch.stack(    [  data[ i+1 : i+1+block_size ]   for i in ix]    )
    
    return x, y


# In[32]:


xb, yb = get_batch("train")


# In[33]:


xb


# In[34]:


yb


# In[35]:


for b in range(batch_size):         
    for t in range(block_size):
        context = xb[b, :t+1]
        target  = yb[b, t]
        print(f"when input is {context.tolist()} the target is: {target}")


# 
# ## Bi-gram model
# 

# In[36]:


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
            
            
            
            


# In[37]:


m   = BigramLanguageModel(vocab_size)

logits, loss = m(xb, yb)

print(logits.shape)



# In[38]:


print(loss)


# In[39]:


logits


# 
# ## Now generate random text before training the BiGram model
# 

# In[40]:


## Kick off generation with some starting token. In this case id 0

idx = torch.zeros(  (1, 1),  dtype=torch.long   )

gen_text = m.generate(idx, max_new_tokens=100)[0].tolist()

print(  decode(gen_text)  )


# 
# ## Training and Optimization
# 

# In[41]:


optimizer = torch.optim.Adam(  m.parameters(), lr=1e-3   )


# In[42]:


batch_size = 32

for step in range(10000):
    xb, yb = get_batch('train')
    
    ## evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)   ## zero out
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(loss.item())


# ## now, regenerate after some training

# In[44]:


## Kick off generation with some starting token. In this case id 0

idx = torch.zeros(  (1, 1),  dtype=torch.long   )

gen_text = m.generate(idx, max_new_tokens=500)[0].tolist()

print(  decode(gen_text)  )









