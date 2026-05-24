import torch

def build_vocab(path):
    words = open(path).read().splitlines()
    chars = sorted(list(set(''.join(words))))
    
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    vocab_size = len(stoi)

    itos = {i: s for s, i in stoi.items()}

    return words, stoi, itos


def build_dataset(words, stoi, context_size):
    X = []
    Y = []

    for w in words:
        context = [0] * context_size

        for ch in w + '.':
            ix = stoi[ch]
    
            X.append(context)
            Y.append(ix)
    
            context = context[1:] + [ix]
    
    return torch.tensor(X), torch.tensor(Y)


def build_splits(words, stoi, context_size, train_split = 80, dev_split = 10, test_split = 10):
    import random
    random.shuffle(words)
    
    n1 = int(train_split / 100 * len(words))
    n2 = int((train_split + dev_split) / 100 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, context_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, context_size)
    Xtest, Ytest = build_dataset(words[n2], stoi, context_size)

    return Xtr, Ytr, Xdev, Ydev, Xtest, Ytest


def softmax(logits, dim = 2):
    counts = logits.exp()
 
    if dim == 2:
        probs = counts / counts.sum(1, keepdim = True)
    else:
        probs = counts / counts.sum()
    
    return probs


def cross_entropy(logits, ys):
    probs = softmax(logits)
    return -probs[torch.arange(ys.nelement()), ys].log().mean()
