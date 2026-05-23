import torch

def build_vocab(path):
    words = open(path).read().splitlines()
    chars = sorted(list(set(''.join(words))))
    
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0

    itos = {i: s for s, i in stoi.items()}

    return words, stoi, itos


def build_dataset(words, stoi, itos):
    X, Y = [], []

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            
            X.append(ix1)
            Y.append(ix2)

    return torch.tensor(X), torch.tensor(Y)


def softmax(logits):
    counts = logits.exp()
    if(counts.ndim >= 2):
        probs = counts / counts.sum(1, keepdim = True)
    else:
        probs = counts / counts.sum()

    return probs


def cross_entropy(logits, ys):
    probs = softmax(logits)
    return -probs[torch.arange(ys.nelement()), ys].log().mean()
