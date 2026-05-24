import torch
from model import MLP
from utils import build_vocab
from train import train

if __name__ == "__main__":
    path = "./../data/names.txt"

    words, stoi, itos = build_vocab(path)
    chars = sorted(list(set(''.join(words))))
    vocab_size = len(stoi)
    
    model = MLP(vocab_size)

    iterations = input("How many iterations for training loop (default 10000): ")
    if iterations == "":
        iterations = 10000
    
    while True:
        train(model, path, iterations)

        response = input("Would you like to continue training? (y/n): ")
        
        if response == 'y':
            iterations = int(input("Number of iterations for the training loop (default = 10000): "))

        else:
            num_samples = int(input("How many samples to generate: "))
            model.sample(num_samples, itos)
            break
