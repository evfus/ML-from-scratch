import torch
from model import MLP
from utils import build_vocab, build_splits

def train(model, path, iterations):
    words, stoi, itos = build_vocab(path)

    Xtr, Ytr, Xdev, Ydev, Xtest, Ytest = build_splits(words, stoi, model.context_size)

    model.train()

    if iterations == "":
        iterations = 10000
    else:
        iterations = int(iterations)

    for step in range(iterations):
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb = Xtr[ix]
        Yb = Ytr[ix]

        logits = model.forward(Xb)
        loss = model.loss(logits, Yb)
        
        for p in model.parameters:
            p.grad = None

        loss.backward()

        lr = 0.01 if step < 2000 else 0.005
        for p in model.parameters:
            p.data += -lr * p.grad

        if step % 1000 == 0:
            print(f'{step} /{iterations}: {loss.item()}')
    
    response = input("Do you want to compare loss between splits? ([y]/n) ")
    if response == 'y' or response == '':
        train_loss = evaluate_split(model, Xtr, Ytr)
        dev_loss = evaluate_split(model, Xdev, Ydev)
        test_loss = evaluate_split(model, Xtest, Ytest)
        
        print(f'\nTraining loss: {train_loss}')
        print(f'Dev loss: {dev_loss}')
        print(f'Test loss: {test_loss}\n')


@torch.no_grad()
def evaluate_split(model, X, Y):
    model.eval()

    logits = model.forward(X)
    loss = model.loss(logits, Y)

    return loss
