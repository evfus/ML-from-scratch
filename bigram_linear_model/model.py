import torch
import matplotlib.pyplot as plt

from utils import softmax, cross_entropy

class BigramLanguageModel:
    def __init__(self, vocab_size):
        self.W = torch.randn((vocab_size, vocab_size), requires_grad = True)
 

    def forward(self, xs):
        logits = self.W[xs]
        probs = softmax(logits)

        return probs


    def loss(self, probs, ys):
        loss = cross_entropy(probs, ys)
        return loss


    def sample(self, stoi, itos, num_samples = 10):
        for i in range(num_samples):
            ix = 0
            letter = ''
            name = ''

            while letter != '.':
                name += letter
                logits = self.W[ix]
                probs = softmax(logits)

                ix = torch.multinomial(probs, num_samples = 1).item()
                letter = itos[ix]
            
            print(f'Sample {i + 1}: {name}')

