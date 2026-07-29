import config
import torch

#dataset loading
f = open('./../data/shakespeare.txt', 'r', encoding = 'utf-8')
text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#mapping for chars
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}

#encode and decode functions
encode = lambda s: [stoi[char] for char in s]
decode = lambda l: ''.join([itos[i] for i in l])

#data splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#batch generator function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.context_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.context_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + config.context_size] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y
