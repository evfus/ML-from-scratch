import config
import dataset
from model import DecoderTransformer
import torch
import matplotlib.pyplot as plt

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


def train(resume):
    model = DecoderTransformer().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    start_iter = 0

    train_losses = []
    val_losses = []
    iterations = []
    
    if resume == True:
        checkpoint = torch.load(
            "outputs/checkpoints/last.pt",
            map_location = config.device
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter"] + 1
        
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        iterations = checkpoint["iterations"]

    end_iter = config.max_iters + start_iter

    for iter in range(start_iter, end_iter):
        if iter % config.eval_interval == 0 or iter == end_iter - 1:
            losses = estimate_loss(model)
            
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            iterations.append(iter)

            print(f'step{iter}: train loss={losses['train']:.4f}, eval loss={losses['val']:.4f}')

        xb, yb = dataset.get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
   
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iter,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "iterations": iterations
        },
        "outputs/checkpoints/last.pt")
    
    plt.plot(iterations, train_losses, label = "Train loss convergence")
    plt.plot(iterations, val_losses, label = "Validation loss convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss.png")
