import torch
from model import MLP
from utils import build_vocab, build_splits

def train(model, path, iterations):
    words, stoi, itos = build_vocab(path)

    Xtr, Ytr, _, _, _, _ = build_splits(words, stoi, model.context_size)
    
    model.train()

    if iterations == "":
        iterations = 10000

    for step in range(iterations):
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb = Xtr[ix]
        Yb = Ytr[ix]

        logits = model.forward(Xb)
        loss = model.loss(logits, Yb)
        
        for p in model.parameters:
            p.grad = None

        loss.backward()

        lr = 1 if step < 100000 else 0.05
        for p in model.parameters:
            p.data += -lr * p.grad

        if step % 200 == 0:
            print(f'{step} /{iterations}: {loss.item()}')
