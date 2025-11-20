#!/usr/bin/env python3
"""
FedGAN PoC Implementation based on Algorithm 1 of the paper.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple, Dict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Reuse models from the tutorial
from train import make_generator_model, make_discriminator_model, generator_loss, discriminator_loss

class Agent:
    """
    Represents a single agent in the FedGAN network.
    Each agent has its own Generator and Discriminator.
    """
    def __init__(self, id: int, dataset: tf.data.Dataset, args: argparse.Namespace):
        self.id = id
        self.dataset = dataset
        self.args = args
        
        # Initialize local models
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        
        # Initialize local optimizers
        # Using same learning rate for both as per PoC plan (can be tuned)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
        
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

    @tf.function
    def train_step(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Algorithm 1: Local update step.
        "Each agent i calculates local stochastic gradient... and updates its local parameter"
        """
        noise = tf.random.normal([tf.shape(images)[0], self.args.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

def prepare_non_iid_datasets(args: argparse.Namespace) -> List[tf.data.Dataset]:
    """
    Splits MNIST into B agents, each getting specific classes to simulate non-IID.
    B=5 agents.
    Agent 0: digits 0, 1
    Agent 1: digits 2, 3
    ...
    """
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

    datasets = []
    num_agents = args.num_agents
    classes_per_agent = 10 // num_agents # Assuming 10 classes and 5 agents -> 2 classes each

    for i in range(num_agents):
        # Filter data for this agent
        start_class = i * classes_per_agent
        end_class = (i + 1) * classes_per_agent
        
        # Create boolean mask
        mask = (train_labels >= start_class) & (train_labels < end_class)
        agent_images = train_images[mask]
        
        print(f"Agent {i}: Classes {start_class}-{end_class-1}, {len(agent_images)} images")
        
        ds = tf.data.Dataset.from_tensor_slices(agent_images)
        ds = ds.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)
        datasets.append(ds)
    
    return datasets

def average_weights(agents: List[Agent]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Algorithm 1: Intermediary calculates average parameters.
    w_n = sum(p_j * w_n^j), theta_n = sum(p_j * theta_n^j)
    Assuming equal p_j for simplicity in this PoC (or proportional to data size if strictly following).
    Since we split MNIST roughly equally (digits are roughly balanced), equal weights is a good approximation.
    """
    # Collect weights
    gen_weights_list = [agent.generator.get_weights() for agent in agents]
    disc_weights_list = [agent.discriminator.get_weights() for agent in agents]
    
    # Average Generator Weights
    avg_gen_weights = []
    for weights_tuple in zip(*gen_weights_list):
        avg_gen_weights.append(np.mean(np.array(weights_tuple), axis=0))
        
    # Average Discriminator Weights
    avg_disc_weights = []
    for weights_tuple in zip(*disc_weights_list):
        avg_disc_weights.append(np.mean(np.array(weights_tuple), axis=0))
        
    return avg_gen_weights, avg_disc_weights

def distribute_weights(agents: List[Agent], avg_gen_weights: List[np.ndarray], avg_disc_weights: List[np.ndarray]):
    """
    Algorithm 1: Intermediary sends back w_n, theta_n and agents update local parameters.
    """
    for agent in agents:
        agent.generator.set_weights(avg_gen_weights)
        agent.discriminator.set_weights(avg_disc_weights)

def generate_and_save_images(model: tf.keras.Model, epoch: int, test_input: tf.Tensor, out_dir: str) -> str:
    # Notice `training` is set to False.
    predictions = model(test_input, training=False)

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

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(path)
    plt.close(fig)
    return path

def make_gif(image_paths: list[str], gif_path: str, duration: float = 0.5) -> None:
    images = []
    for p in image_paths:
        images.append(imageio.v2.imread(p))
    imageio.mimsave(gif_path, images, duration=duration)

def plot_and_save_losses(loss_history: pd.DataFrame, out_png: str, out_csv: str) -> None:
    loss_history.to_csv(out_csv, index=False)

    plt.figure()
    plt.plot(loss_history['epoch'], loss_history['gen_loss'], label='gen_loss')
    plt.plot(loss_history['epoch'], loss_history['disc_loss'], label='disc_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png)
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FedGAN PoC')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=60000, help='Shuffle buffer size')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of generator noise vector')
    parser.add_argument('--num_examples_to_generate', type=int, default=16, help='Number of images to generate per epoch')
    parser.add_argument('--output_dir', default='fedgan_outputs', help='Directory to save outputs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
    parser.add_argument('--sync_interval', type=int, default=20, help='Synchronization interval (K)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Setup output directories
    out_dir = args.output_dir
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare Datasets
    print("Preparing Non-IID Datasets...")
    agent_datasets = prepare_non_iid_datasets(args)
    
    # Initialize Agents
    print(f"Initializing {args.num_agents} Agents...")
    agents = [Agent(i, ds, args) for i, ds in enumerate(agent_datasets)]
    
    # Visualization seed
    fixed_seed = tf.random.normal([args.num_examples_to_generate, args.noise_dim])
    
    # Training Loop
    print("Starting Training...")
    loss_history = []
    image_paths = []
    
    global_step = 0
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Reset metrics
        for agent in agents:
            agent.gen_loss_metric.reset_state()
            agent.disc_loss_metric.reset_state()
            
        # Create iterators for all agents
        # We iterate until the largest dataset is exhausted, or shortest?
        # Usually in FedAvg, we assume rounds. Here we simulate "epochs".
        # Since datasets are roughly equal, we can zip them or iterate max steps.
        # Let's iterate based on the number of batches in the first agent (approx equal).
        # Or better, just iterate through a fixed number of steps per epoch if sizes differ.
        # Since we split MNIST evenly (approx 12000 images per agent), they should have similar batches.
        
        # We will iterate through the datasets. Since they might have slightly different sizes,
        # we can use zip_longest or just iterate until one finishes.
        # For simplicity in PoC, we'll iterate through the minimum length.
        
        iterators = [iter(agent.dataset) for agent in agents]
        steps_per_epoch = min([len(ds) for ds in agent_datasets]) # This might be slow to calc if not eager?
        # len(ds) works for tensor_slices dataset.
        
        for step in range(steps_per_epoch):
            global_step += 1
            
            # 1. Local Update
            for i, agent in enumerate(agents):
                try:
                    batch = next(iterators[i])
                    g_loss, d_loss = agent.train_step(batch)
                    agent.gen_loss_metric.update_state(g_loss)
                    agent.disc_loss_metric.update_state(d_loss)
                except StopIteration:
                    pass # Should not happen if we use range(steps_per_epoch)
            
            # 2. Synchronization (Algorithm 1: If n mod K == 0)
            if global_step % args.sync_interval == 0:
                # print(f"  Syncing at step {global_step}")
                avg_gen, avg_disc = average_weights(agents)
                distribute_weights(agents, avg_gen, avg_disc)
        
        # End of Epoch
        # Record average losses across all agents
        avg_gen_loss = np.mean([agent.gen_loss_metric.result() for agent in agents])
        avg_disc_loss = np.mean([agent.disc_loss_metric.result() for agent in agents])
        
        print(f"  Losses: Gen={avg_gen_loss:.4f}, Disc={avg_disc_loss:.4f}, Time={time.time()-epoch_start:.2f}s")
        loss_history.append({'epoch': epoch, 'gen_loss': avg_gen_loss, 'disc_loss': avg_disc_loss})
        
        # Generate images using Agent 0's generator (they should be synced/similar)
        # Or better, use the averaged weights? 
        # The paper says "The intermediary send back w_n, theta_n and agents update local parameters".
        # So at the end of sync, everyone is same. 
        # But if epoch ends NOT on a sync step, they might differ slightly.
        # We'll use Agent 0 as representative.
        img_path = generate_and_save_images(agents[0].generator, epoch, fixed_seed, images_dir)
        image_paths.append(img_path)
        
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s")
    
    # Save Outputs
    losses_png = os.path.join(out_dir, 'losses.png')
    losses_csv = os.path.join(out_dir, 'losses.csv')
    plot_and_save_losses(pd.DataFrame(loss_history), losses_png, losses_csv)
    
    gif_path = os.path.join(out_dir, 'training_progress.gif')
    make_gif(image_paths, gif_path)
    print(f"Outputs saved to {out_dir}")

if __name__ == '__main__':
    main()
