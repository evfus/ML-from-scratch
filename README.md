# ML from Scratch

A collection of machine learning and deep learning projects implemented from scratch while learning the fundamentals of neural networks.

The goal of this repository is to understand how modern models work by implementing them myself instead of relying on high-level APIs. Most of the code uses PyTorch tensors and autograd, but the model architectures and training logic are implemented manually.

## Contents

### Bigram Linear Language Model

A simple character-level language model trained on a dataset of names.

Features:
- Character embeddings
- Bigram probability matrix
- Cross-entropy loss
- Sampling new names after training

Directory:
```
bigram_linear_model/
```

---

### Bigram MLP Language Model

An improved character-level language model based on a multilayer perceptron.

Compared to the linear model, this implementation introduces:
- Learnable embeddings
- Hidden layers
- Batch Normalization
- Tanh activations
- Better weight initialization
- Training diagnostics and visualizations

The implementation includes custom versions of several neural network components to better understand how they work internally instead of using `torch.nn` modules.

Directory:
```
bigram_MLP_model/
```

## Repository Structure

```
.
├── bigram_linear_model/
├── bigram_MLP_model/
├── data/
│   ├── names.txt
│   └── shakespeare.txt
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- matplotlib

Install the required packages:

```bash
pip install torch matplotlib
```

## Running the Projects

### Bigram Linear Model

```bash
cd bigram_linear_model
python main.py
```

### Bigram MLP Model

```bash
cd bigram_MLP_model
python main.py
```

The programs will ask for the number of training iterations before starting training.

## Training Analysis

The MLP implementation includes several plots that help analyse training behaviour:

- Activation distributions
- Gradient distributions
- Update-to-data ratio

These are saved in:

```
bigram_MLP_model/outputs/
```

## What I Learned

Working on these projects helped me better understand:

- Gradient descent and backpropagation
- Character embeddings
- Cross-entropy loss
- Weight initialization
- Batch Normalization
- Activation functions
- Building neural network layers from scratch
- Training and debugging neural networks
