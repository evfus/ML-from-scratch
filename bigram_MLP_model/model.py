import torch
from utils import softmax, cross_entropy

class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = (torch.randn((fan_in, fan_out)) / fan_in ** 0.5) * 5/3
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias
        
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gain = torch.ones(dim)
        self.bias = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim = True)
            xvar = x.var(0, keepdim = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        self.out = ((x - xmean) / torch.sqrt(xvar + self.eps)) * self.gain + self.bias
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gain, self.bias]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)

        if torch.is_grad_enabled():
            self.out.retain_grad()

        return self.out

    def parameters(self):
        return []


class MLP:
    def __init__(self, vocab_size, context_size = 3, emb_dim = 10, n_hidden = 300):
        self.embeddings = torch.randn((vocab_size, emb_dim))
        self.context_size = context_size

        self.layers = [
            Linear(emb_dim * self.context_size, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
            Linear(n_hidden, vocab_size, bias = False), BatchNorm1d(vocab_size)
        ]

        self.parameters = [self.embeddings] + [p for layer in self.layers for p in layer.parameters()]

        self.training = True

        for p in self.parameters:
            p.requires_grad = True

    def forward(self, x):
        emb = self.embeddings[x]
        x = emb.view(emb.shape[0], -1)

        for layer in self.layers:
            x = layer(x)

        return x

    def loss(self, logits, Yb):
        return cross_entropy(logits, Yb)
   
    def sample(self, num_samples, itos):
        if self.training == True:
            self.eval()

        for i in range(num_samples):
            name = []
            context = [0] * self.context_size

            while True:
                logits = self.forward(torch.tensor([context]))
                probs = softmax(logits, dim = 1)

                ix = torch.multinomial(probs, num_samples = 1).item()
                if ix != 0:
                    name.append(ix)
                    context = context[1:] + [ix]
                else:
                    break

            print(''.join(itos[i] for i in name))

    def eval(self):
        self.training = False

        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False

    def train(self):
        self.training = True

        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True
