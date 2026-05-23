from model import BigramLanguageModel
from train import train
from utils import build_vocab

if __name__ == "__main__":
    path = "././data/names.txt"
    iterations = int(input("How many iterations for the training loop (default = 10000): "))

    words, stoi, itos = build_vocab(path)
    chars = sorted(list(set(''.join(words))))
    vocab_size = len(chars) + 1

    model = BigramLanguageModel(vocab_size)

    while True:
        train(model, path, iterations)
        response = input("Would you like to continue training? (y/n): ")
        
        if response == 'y':
            iterations = int(input("Number of iterations for the training loop (default = 10000): "))

        else:
            num_samples = int(input("How many samples to generate: "))
            model.sample(stoi, itos, num_samples)
            break


