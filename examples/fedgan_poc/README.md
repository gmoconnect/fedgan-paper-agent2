# FedGAN Proof of Concept

Implementation of Federated Generative Adversarial Network (FedGAN) based on the paper's Algorithm 1.

## Features

- Non-IID MNIST data partitioning across 5 agents (2 classes per agent)
- Periodic parameter synchronization (K=20 iterations)
- Weighted averaging based on local dataset sizes
- Per-epoch GIF generation showing generator evolution
- Combined generator/discriminator loss curves

## Setup

1. Create virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

## Output

- `outputs/images/` - Generated images per epoch and final animation GIF
- `outputs/checkpoints/` - Model checkpoints at synchronization points
- `outputs/metrics/` - Loss curves and CSV data

## Architecture

- **B=5 agents**, each with local Generator and Discriminator
- **K=20** synchronization interval
- **50 epochs** training
- **Batch size**: 64
- **Learning rate**: 1e-4 (Adam with beta_1=0.5)
- **Noise dimension**: 100

Based on paper: arXiv:2006.07228v2
