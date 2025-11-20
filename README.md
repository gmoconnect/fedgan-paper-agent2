# FedGAN PoC Implementation

Proof of Concept implementation of **FedGAN (Federated Generative Adversarial Network)** based on Algorithm 1 from the paper:

> **"FedGAN: Federated Generative Adversarial Networks for Distributed Data"**  
> Mohammad Rasouli, Tao Sun, Ram Rajagopal  
> arXiv:2006.07228v2

## Overview

This PoC implements the core FedGAN algorithm with:
- **5 agents**, each with local generator and discriminator
- **Non-IID MNIST dataset** partitioned by digit classes (each agent gets 2 digit classes)
- **Single-process simulation** (no actual network communication)
- **Synchronization every K=20 iterations** via weighted parameter averaging
- **Visualization outputs**: animated GIF of generated images, loss graphs

## Algorithm 1 Mapping

The implementation closely follows Algorithm 1 from the paper:

1. **Initialization** (Line 166): All agents initialize with same parameters
2. **Main Loop** (Line 169): Training for N epochs
3. **Local Gradient Calculation** (Line 170): Each agent computes stochastic gradients from local data
4. **Local Parameter Update** (Lines 176-177, Eq. 1): SGD updates for discriminator and generator
5. **Synchronization Check** (Line 181): Every K iterations
6. **Parameter Averaging** (Lines 184-188, Eq. 2): Weighted averaging at server
7. **Parameter Broadcasting** (Lines 191-192): Sync all agents with averaged parameters

See detailed comments in `fedgan_poc.py` for line-by-line mapping.

## Setup

### Prerequisites

- Python 3.12
- macOS (for TensorFlow Metal GPU support)

### Installation

1. Create and activate virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the FedGAN PoC:

```bash
python fedgan_poc.py
```

This will:
- Load and partition MNIST dataset across 5 agents (non-IID)
- Train FedGAN for 50 epochs with synchronization every 20 iterations
- Generate outputs in `outputs/` directory

## Configuration

Edit `config.py` to modify parameters:

- `NUM_AGENTS` (B): Number of agents (default: 5)
- `NUM_EPOCHS`: Training epochs (default: 50)
- `SYNC_INTERVAL` (K): Synchronization interval (default: 20)
- `LEARNING_RATE_DISCRIMINATOR` (a(n)): Discriminator learning rate (default: 1e-4)
- `LEARNING_RATE_GENERATOR` (b(n)): Generator learning rate (default: 1e-4)
- `BATCH_SIZE`: Batch size (default: 256)
- `DIGITS_PER_AGENT`: Digit classes per agent for non-IID split (default: 2)

## Outputs

After training, the following files are generated in `outputs/`:

- **`training_progress.gif`**: Animated GIF showing generated images from Agent 0 across all epochs
- **`losses.png`**: Graph of generator and discriminator losses (Agent 0) per epoch
- **`losses.csv`**: CSV file with loss data
- **`images/`**: Directory containing per-epoch generated images

## Architecture

### File Structure

```
fedgan-paper-agent2/
├── fedgan_poc.py          # Main FedGAN implementation (Algorithm 1)
├── models.py              # GAN model definitions (DCGAN from TF tutorial)
├── data_utils.py          # MNIST loading and non-IID partitioning
├── visualization.py       # Image generation, GIF creation, loss plotting
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Key Classes

- **`FedGANAgent`**: Represents a single agent with local generator, discriminator, and dataset
- **`FedGANServer`**: Intermediary server for weighted parameter averaging
- **`FedGANTrainer`**: Orchestrates the training loop and synchronization

## Implementation Notes

1. **Single-Process Simulation**: All agents run in the same process sequentially (not parallel). In a real federated setting, agents would train in parallel.

2. **Non-IID Data Split**: MNIST is partitioned by digit classes:
   - Agent 0: digits 0, 1
   - Agent 1: digits 2, 3
   - Agent 2: digits 4, 5
   - Agent 3: digits 6, 7
   - Agent 4: digits 8, 9

3. **Weighted Averaging**: Agent weights (p_j) are calculated based on local dataset sizes, as specified in Algorithm 1, Equation 2.

4. **Visualization**: Only Agent 0's outputs are visualized (generated images and losses), as per user requirements.

5. **Equal Time-Scale**: Uses equal learning rates for generator and discriminator (a(n) = b(n) = 1e-4).

## References

- Paper: [arXiv:2006.07228v2](https://arxiv.org/abs/2006.07228)
- TensorFlow DCGAN Tutorial: [https://www.tensorflow.org/tutorials/generative/dcgan](https://www.tensorflow.org/tutorials/generative/dcgan)

## License

See LICENSE file.
