# DCGAN tutorial CLI

This folder contains a CLI implementation of the TensorFlow DCGAN tutorial.

Requirements
- Python 3.12 (use a venv)
- See `requirements.txt` for the Python packages. This includes TensorFlow 2.16.2 and tensorflow-metal for macOS GPU support.

Quick start

1. Create venv and activate (zsh):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Run training (example):

```bash
python train.py --epochs 50 --batch_size 256 --output_dir outputs --save_gif
```

Outputs
- images/: per-epoch PNGs named `image_at_epoch_XXXX.png`
- training_progress.gif: optional animated GIF of epoch images (if --save_gif)
- losses.png: PNG plot of generator and discriminator losses
- losses.csv: CSV with columns `epoch,gen_loss,disc_loss`

Notes
- Model, loss, and optimizers follow the TensorFlow tutorial.
- The visualization seed is fixed (CLI `--seed`) and only used for generating the per-epoch images and GIF; batch-level randomness for training uses random noise per batch.
