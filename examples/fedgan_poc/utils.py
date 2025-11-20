"""
Utility functions for FedGAN training.

Provides functions for:
- Image generation and visualization
- GIF animation creation
- Loss curve plotting
- Checkpoint management
"""

import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import pandas as pd
from typing import List, Tuple


def generate_and_save_images(
    generator: tf.keras.Model,
    epoch: int,
    test_input: tf.Tensor,
    save_dir: str = 'outputs/images'
) -> str:
    """
    Generate images from fixed noise and save to file.
    
    Args:
        generator: Generator model
        epoch: Current epoch number
        test_input: Fixed noise vector for consistent visualization
        save_dir: Directory to save images
    
    Returns:
        Path to saved image file
    """
    # Generate images
    predictions = generator(test_input, training=False)
    
    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(8, 8))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Rescale from [-1, 1] to [0, 1]
        img = (predictions[i, :, :, 0] + 1) / 2.0
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save image
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'image_epoch_{epoch:04d}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


def create_animation_gif(
    image_dir: str = 'outputs/images',
    output_path: str = 'outputs/images/training_animation.gif',
    duration: float = 0.5
) -> None:
    """
    Create animated GIF from generated images across epochs.
    
    Args:
        image_dir: Directory containing epoch images
        output_path: Path to save output GIF
        duration: Duration of each frame in seconds
    """
    # Find all epoch images
    image_files = sorted(glob.glob(os.path.join(image_dir, 'image_epoch_*.png')))
    
    if not image_files:
        print(f"Warning: No images found in {image_dir}")
        return
    
    # Load images
    images = []
    for filename in image_files:
        img = Image.open(filename)
        images.append(img)
    
    # Save as GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(
        output_path,
        images,
        duration=duration,
        loop=0  # Infinite loop
    )
    
    print(f"Animation saved to {output_path} ({len(images)} frames)")


def plot_losses(
    losses_history: List[Tuple[int, float, float]],
    save_path: str = 'outputs/metrics/losses.png'
) -> None:
    """
    Plot generator and discriminator losses over epochs.
    
    Args:
        losses_history: List of (epoch, gen_loss, disc_loss) tuples
        save_path: Path to save plot
    """
    epochs, gen_losses, disc_losses = zip(*losses_history)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, gen_losses, label='Generator Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, disc_losses, label='Discriminator Loss', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('FedGAN Training Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Loss plot saved to {save_path}")


def save_losses_csv(
    losses_history: List[Tuple[int, float, float]],
    save_path: str = 'outputs/metrics/losses.csv'
) -> None:
    """
    Save loss history to CSV file.
    
    Args:
        losses_history: List of (epoch, gen_loss, disc_loss) tuples
        save_path: Path to save CSV
    """
    df = pd.DataFrame(
        losses_history,
        columns=['epoch', 'generator_loss', 'discriminator_loss']
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Loss data saved to {save_path}")


def aggregate_model_weights(
    models: List[tf.keras.Model],
    weights: np.ndarray
) -> List[np.ndarray]:
    """
    Compute weighted average of model parameters.
    
    Implements Algorithm 1 Step 3:
    w_n = sum_j (p_j * w_n_j)
    theta_n = sum_j (p_j * theta_n_j)
    
    Args:
        models: List of models from all agents
        weights: Array of aggregation weights p_i (should sum to 1.0)
    
    Returns:
        List of averaged weight arrays
    """
    # Get number of layers
    num_layers = len(models[0].get_weights())
    
    # Initialize aggregated weights
    aggregated_weights = []
    
    for layer_idx in range(num_layers):
        # Weighted sum across agents
        layer_weight = None
        for agent_idx, model in enumerate(models):
            agent_weight = model.get_weights()[layer_idx]
            
            if layer_weight is None:
                layer_weight = weights[agent_idx] * agent_weight
            else:
                layer_weight += weights[agent_idx] * agent_weight
        
        aggregated_weights.append(layer_weight)
    
    return aggregated_weights


def synchronize_models(
    models: List[tf.keras.Model],
    aggregated_weights: List[np.ndarray]
) -> None:
    """
    Update all agent models with aggregated weights.
    
    Implements Algorithm 1 Step 3:
    w_n_i = w_n for all agents i
    theta_n_i = theta_n for all agents i
    
    Args:
        models: List of models to update
        aggregated_weights: Averaged weights to broadcast
    """
    for model in models:
        model.set_weights(aggregated_weights)


def save_checkpoint(
    generators: List[tf.keras.Model],
    discriminators: List[tf.keras.Model],
    gen_optimizers: List[tf.keras.optimizers.Optimizer],
    disc_optimizers: List[tf.keras.optimizers.Optimizer],
    epoch: int,
    checkpoint_dir: str = 'outputs/checkpoints'
) -> None:
    """
    Save model checkpoints for all agents.
    
    Args:
        generators: List of generator models
        discriminators: List of discriminator models
        gen_optimizers: List of generator optimizers
        disc_optimizers: List of discriminator optimizers
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save only the first agent's models (they are synchronized)
    checkpoint = tf.train.Checkpoint(
        generator=generators[0],
        discriminator=discriminators[0],
        generator_optimizer=gen_optimizers[0],
        discriminator_optimizer=disc_optimizers[0]
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch:04d}')
    checkpoint.save(file_prefix=checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing FedGAN utilities...")
    print("=" * 60)
    
    # Test weight aggregation
    print("\nTesting weight aggregation...")
    from models import make_generator
    
    # Create 3 dummy generators
    gens = [make_generator(100) for _ in range(3)]
    
    # Dummy weights that sum to 1.0
    weights = np.array([0.2, 0.3, 0.5])
    
    # Aggregate
    agg_weights = aggregate_model_weights(gens, weights)
    print(f"Aggregated {len(agg_weights)} layers")
    
    # Synchronize
    synchronize_models(gens, agg_weights)
    print("Models synchronized")
    
    # Verify all models have same weights
    w0 = gens[0].get_weights()[0]
    w1 = gens[1].get_weights()[0]
    w2 = gens[2].get_weights()[0]
    
    print(f"Models equal after sync: {np.allclose(w0, w1) and np.allclose(w1, w2)}")
    
    print("\n" + "=" * 60)
    print("Utilities test complete!")
