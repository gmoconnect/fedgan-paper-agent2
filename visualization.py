"""Visualization utilities for FedGAN PoC

Generates images, GIFs, and loss plots.
"""
import os
from typing import List
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def generate_and_save_images(
    generator: tf.keras.Model,
    epoch: int,
    test_input: tf.Tensor,
    output_dir: str
) -> str:
    """Generate and save images from generator
    
    Args:
        generator: Generator model
        epoch: Current epoch number
        test_input: Fixed noise vector for consistent visualization
        output_dir: Directory to save images
        
    Returns:
        Path to saved image
    """
    # Generate images (inference mode)
    predictions = generator(test_input, training=False)
    
    # Rescale from [-1,1] to [0,255]
    imgs = (predictions.numpy() * 127.5 + 127.5).astype(np.uint8)
    imgs = imgs.reshape((-1, 28, 28))
    
    n = int(np.ceil(np.sqrt(imgs.shape[0])))
    fig = plt.figure(figsize=(n, n))
    
    for i in range(imgs.shape[0]):
        plt.subplot(n, n, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}', fontsize=12)
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(path)
    plt.close(fig)
    return path


def create_animation_gif(
    image_paths: List[str],
    gif_path: str,
    duration: float = 0.5
) -> None:
    """Create animated GIF from image sequence
    
    Args:
        image_paths: List of paths to images
        gif_path: Output path for GIF
        duration: Duration per frame in seconds
    """
    images = []
    for p in image_paths:
        images.append(imageio.v2.imread(p))
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved to {gif_path}")


def plot_and_save_losses(
    loss_history: pd.DataFrame,
    output_png: str,
    output_csv: str
) -> None:
    """Plot and save loss history
    
    Creates a single graph with both generator and discriminator losses.
    
    Args:
        loss_history: DataFrame with columns: epoch, gen_loss, disc_loss
        output_png: Output path for PNG graph
        output_csv: Output path for CSV data
    """
    # Save CSV
    loss_history.to_csv(output_csv, index=False)
    print(f"Loss CSV saved to {output_csv}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['epoch'], loss_history['gen_loss'], label='Generator Loss', marker='o')
    plt.plot(loss_history['epoch'], loss_history['disc_loss'], label='Discriminator Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FedGAN Training Losses (Agent 0)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_png)
    plt.close()
    print(f"Loss graph saved to {output_png}")
