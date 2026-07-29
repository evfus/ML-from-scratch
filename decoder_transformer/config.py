import torch

batch_size = 32
context_size = 8
max_iters = 10000
eval_interval = 50
learning_rate = 1e-3
eval_iters = 100

n_embd = 32
n_heads = 4
n_layers = 4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
