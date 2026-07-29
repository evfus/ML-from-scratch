import config
import dataset
from model import DecoderTransformer
import torch

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = dataset.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train():
    model = DecoderTransformer()
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model)
            print(f'step{iter}: train loss={losses['train']:.4f}, eval loss={losses['val']:.4f}')

        xb, yb = dataset.get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    
    context = torch.zeros((1, 1), dtype = torch.long, device = config.device)
    print(dataset.decode(model.generate(context, 2000)[0].tolist()))
