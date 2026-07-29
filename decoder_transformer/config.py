batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

n_embd = 384
n_heads = 6
n_layers = 6
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
