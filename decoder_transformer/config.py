import torch

batch_size = 64
context_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 50

n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
