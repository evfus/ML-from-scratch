import torch
import matplotlib.pyplot as plt
from model import Tanh

def plot_activation_distribution(layers, path="./outputs/activation_distribution.png"):
    plt.figure(figsize = (20, 4))

    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out
            hy, hx = torch.histogram(t, density = True)
            plt.plot(hx[:-1].detach(), hy.detach(), label = f"layer {i}")
 
    plt.title("Activation distributions")
    plt.legend()
    
    plt.savefig(path)


def plot_gradient_distribution(layers, path = "./outputs/gradient_distribution.png"):
    plt.figure(figsize = (20, 4))

    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            g = layer.out.grad
            hy, hx = torch.histogram(g, density = True)
            plt.plot(hx[:-1].detach(), hy.detach(), label = f"layer {i}")

    plt.title("Gradient distributions")
    plt.legend()
  
    plt.savefig(path)


def plot_update_data_ratio(parameters, ud, path = "./outputs/update_to_data_ratio.png"):
    plt.figure(figsize = (20, 4))

    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))], label = f"param {i}")
    
    plt.plot([0, len(ud)], [-3, -3], color='k')
    
    plt.title("Update to data ratio")
    plt.legend()
    
    plt.savefig(path)

