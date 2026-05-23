import torch
from model import BigramLanguageModel
from utils import build_vocab, build_dataset
#import matplotlib.pyplot as plt


def train(model, path, iterations):
    words, stoi, itos = build_vocab(path)
    xs, ys = build_dataset(words, stoi, itos)
    
    if iterations == "":
        iterations = 10000

    for step in range(iterations):
        logits = model.forward(xs)
        loss = model.loss(logits, ys)

        model.W.grad = None
        loss.backward()
        
        lr = 20 if step > 200 else 50
        model.W.data -= lr * model.W.grad

        if step % 50 == 0:
            print(loss.item())

