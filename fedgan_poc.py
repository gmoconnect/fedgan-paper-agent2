#!/usr/bin/env python3
"""FedGAN PoC Implementation

Implements Algorithm 1: Federated Generative Adversarial Network (FedGAN)
from the paper "FedGAN: Federated Generative Adversarial Networks for Distributed Data"
"""
import os
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from models import (
    make_generator_model,
    make_discriminator_model,
    generator_loss,
    discriminator_loss
)
from data_utils import load_and_partition_mnist
from visualization import (
    generate_and_save_images,
    create_animation_gif,
    plot_and_save_losses
)


class FedGANAgent:
    """FedGAN Agent with local generator and discriminator
    
    Algorithm 1: Each agent i has:
    - Local dataset R_i
    - Local discriminator with parameters w_i
    - Local generator with parameters theta_i
    - Local optimizers for SGD updates
    """
    
    def __init__(
        self,
        agent_id: int,
        dataset: tf.data.Dataset,
        noise_dim: int,
        learning_rate_disc: float,
        learning_rate_gen: float
    ):
        """Initialize FedGAN agent
        
        Algorithm 1 Input: Initialize w_0^i = w_hat, theta_0^i = theta_hat
        
        Args:
            agent_id: Agent identifier
            dataset: Local dataset R_i
            noise_dim: Dimension of noise vector for generator
            learning_rate_disc: Learning rate a(n) for discriminator
            learning_rate_gen: Learning rate b(n) for generator
        """
        self.agent_id = agent_id
        self.dataset = dataset
        self.noise_dim = noise_dim
        
        # Algorithm 1: Local discriminator and generator
        self.discriminator = make_discriminator_model()
        self.generator = make_generator_model()
        
        # Algorithm 1: Learning rates a(n) and b(n)
        # Using Adam optimizer with specified learning rates
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_disc,
            beta_1=0.5
        )
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_gen,
            beta_1=0.5
        )
    
    @tf.function
    def train_step(self, real_images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Single training step for local agent
        
        Algorithm 1, Lines 170-179:
        1. Calculate local stochastic gradients g_tilde_i and h_tilde_i
        2. Update local parameters w_n^i and theta_n^i
        
        Args:
            real_images: Batch of real images from local dataset R_i
            
        Returns:
            gen_loss: Generator loss
            disc_loss: Discriminator loss
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        # Algorithm 1, Line 170: Calculate local stochastic gradients
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data with local generator
            generated_images = self.generator(noise, training=True)
            
            # Discriminator outputs on real and fake data
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate losses (stochastic gradients)
            # g_tilde_i: discriminator gradient from R_i
            disc_loss = discriminator_loss(real_output, fake_output)
            
            # h_tilde_i: generator gradient from R_i and fake data
            gen_loss = generator_loss(fake_output)
        
        # Compute gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )
        
        # Algorithm 1, Lines 176-177 (Equation 1): Update local parameters
        # w_n^i = w_(n-1)^i + a(n-1) * g_tilde_i
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        
        # theta_n^i = theta_(n-1)^i + b(n-1) * h_tilde_i
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get current model parameters
        
        Algorithm 1, Line 182: Agents send parameters to intermediary
        
        Returns:
            discriminator_params: List of discriminator weight arrays
            generator_params: List of generator weight arrays
        """
        disc_params = [w.numpy() for w in self.discriminator.trainable_variables]
        gen_params = [w.numpy() for w in self.generator.trainable_variables]
        return disc_params, gen_params
    
    def set_parameters(
        self,
        discriminator_params: List[np.ndarray],
        generator_params: List[np.ndarray]
    ) -> None:
        """Set model parameters from server
        
        Algorithm 1, Lines 191-192: Agents update local parameters
        w_n^i = w_n, theta_n^i = theta_n
        
        Args:
            discriminator_params: Averaged discriminator parameters from server
            generator_params: Averaged generator parameters from server
        """
        for var, param in zip(self.discriminator.trainable_variables, discriminator_params):
            var.assign(param)
        for var, param in zip(self.generator.trainable_variables, generator_params):
            var.assign(param)


class FedGANServer:
    """FedGAN Intermediary Server
    
    Algorithm 1: Intermediary averages and broadcasts parameters
    """
    
    @staticmethod
    def aggregate_parameters(
        agent_params: List[Tuple[List[np.ndarray], List[np.ndarray]]],
        weights: List[float]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Aggregate agent parameters using weighted averaging
        
        Algorithm 1, Lines 184-188 (Equation 2):
        w_n = sum(p_j * w_n^j) for j=1 to B
        theta_n = sum(p_j * theta_n^j) for j=1 to B
        
        Args:
            agent_params: List of (disc_params, gen_params) from each agent
            weights: Agent weights p_j (data size ratios)
            
        Returns:
            avg_disc_params: Averaged discriminator parameters
            avg_gen_params: Averaged generator parameters
        """
        num_agents = len(agent_params)
        
        # Initialize averaged parameters with zeros
        disc_params_0, gen_params_0 = agent_params[0]
        avg_disc_params = [np.zeros_like(p) for p in disc_params_0]
        avg_gen_params = [np.zeros_like(p) for p in gen_params_0]
        
        # Weighted sum: sum(p_j * params_j)
        for (disc_params, gen_params), weight in zip(agent_params, weights):
            for i, param in enumerate(disc_params):
                avg_disc_params[i] += weight * param
            for i, param in enumerate(gen_params):
                avg_gen_params[i] += weight * param
        
        return avg_disc_params, avg_gen_params


class FedGANTrainer:
    """FedGAN Training Orchestrator
    
    Implements the main training loop from Algorithm 1
    """
    
    def __init__(
        self,
        num_agents: int,
        num_epochs: int,
        sync_interval: int,
        noise_dim: int,
        learning_rate_disc: float,
        learning_rate_gen: float,
        output_dir: str
    ):
        """Initialize FedGAN trainer
        
        Args:
            num_agents: Number of agents B
            num_epochs: Number of training epochs
            sync_interval: Synchronization interval K
            noise_dim: Noise dimension for generator
            learning_rate_disc: Discriminator learning rate a(n)
            learning_rate_gen: Generator learning rate b(n)
            output_dir: Output directory for results
        """
        self.num_agents = num_agents
        self.num_epochs = num_epochs
        self.sync_interval = sync_interval
        self.noise_dim = noise_dim
        self.learning_rate_disc = learning_rate_disc
        self.learning_rate_gen = learning_rate_gen
        self.output_dir = output_dir
        
        self.agents: List[FedGANAgent] = []
        self.server = FedGANServer()
        self.agent_weights: List[float] = []
        
        # For visualization (first agent only)
        self.fixed_seed = None
        self.image_paths: List[str] = []
        self.loss_history: List[dict] = []
    
    def setup(
        self,
        datasets: List[tf.data.Dataset],
        weights: List[float]
    ) -> None:
        """Setup agents with datasets
        
        Algorithm 1 Input: Initialize agents
        
        Args:
            datasets: List of datasets, one per agent
            weights: Agent weights p_j
        """
        self.agent_weights = weights
        
        # Create agents
        for agent_id, dataset in enumerate(datasets):
            agent = FedGANAgent(
                agent_id=agent_id,
                dataset=dataset,
                noise_dim=self.noise_dim,
                learning_rate_disc=self.learning_rate_disc,
                learning_rate_gen=self.learning_rate_gen
            )
            self.agents.append(agent)
        
        # Fixed seed for visualization (first agent)
        np.random.seed(config.SEED)
        self.fixed_seed = tf.convert_to_tensor(
            np.random.normal(size=(config.NUM_EXAMPLES_TO_GENERATE, self.noise_dim)).astype(np.float32)
        )
        
        print(f"Setup complete: {len(self.agents)} agents initialized\n")
    
    def train(self) -> None:
        """Main training loop
        
        Algorithm 1, Lines 169-195: Main loop for n=1 to N-1
        """
        print("Starting FedGAN training...")
        print(f"Epochs: {self.num_epochs}, Sync Interval: {self.sync_interval}\n")
        
        start_time = time.time()
        iteration = 0
        
        # Algorithm 1, Line 169: For n=1,2,...,N-1
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            
            # Track losses for first agent only
            epoch_gen_losses = []
            epoch_disc_losses = []
            
            # Train each agent on their local data
            # Note: In real federated setting, this would be parallel
            # Here we simulate sequentially
            for agent in self.agents:
                for batch in agent.dataset:
                    # Algorithm 1, Lines 170-179: Local training step
                    gen_loss, disc_loss = agent.train_step(batch)
                    
                    # Track first agent's losses
                    if agent.agent_id == 0:
                        epoch_gen_losses.append(float(gen_loss.numpy()))
                        epoch_disc_losses.append(float(disc_loss.numpy()))
                    
                    iteration += 1
                    
                    # Algorithm 1, Line 181: If n mod K == 0 (synchronization)
                    if iteration % self.sync_interval == 0:
                        self._synchronize_agents()
                        print(f"  Iteration {iteration}: Synchronized agents")
            
            # Compute epoch average losses (first agent)
            avg_gen_loss = np.mean(epoch_gen_losses) if epoch_gen_losses else 0.0
            avg_disc_loss = np.mean(epoch_disc_losses) if epoch_disc_losses else 0.0
            
            print(f"  Agent 0 - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            
            # Save loss history
            self.loss_history.append({
                'epoch': epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss
            })
            
            # Generate and save images (first agent only)
            img_path = generate_and_save_images(
                self.agents[0].generator,
                epoch,
                self.fixed_seed,
                os.path.join(self.output_dir, 'images')
            )
            self.image_paths.append(img_path)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.2f}s")
        
        # Save results
        self._save_results()
    
    def _synchronize_agents(self) -> None:
        """Synchronize agent parameters via server
        
        Algorithm 1, Lines 182-193:
        1. Agents send parameters to intermediary
        2. Intermediary calculates averaged parameters
        3. Intermediary sends back averaged parameters
        4. Agents update local parameters
        """
        # Algorithm 1, Line 182: All agents send parameters to intermediary
        agent_params = [agent.get_parameters() for agent in self.agents]
        
        # Algorithm 1, Lines 184-188 (Equation 2): Server aggregates
        avg_disc_params, avg_gen_params = self.server.aggregate_parameters(
            agent_params,
            self.agent_weights
        )
        
        # Algorithm 1, Lines 189-193: Broadcast and update
        for agent in self.agents:
            agent.set_parameters(avg_disc_params, avg_gen_params)
    
    def _save_results(self) -> None:
        """Save training results (GIF, loss plots)"""
        # Create GIF
        gif_path = os.path.join(self.output_dir, config.GIF_FILENAME)
        create_animation_gif(self.image_paths, gif_path, config.GIF_DURATION)
        
        # Save loss plots
        loss_df = pd.DataFrame(self.loss_history)
        loss_png = os.path.join(self.output_dir, config.LOSS_PNG_FILENAME)
        loss_csv = os.path.join(self.output_dir, config.LOSS_CSV_FILENAME)
        plot_and_save_losses(loss_df, loss_png, loss_csv)


def main():
    """Main entry point for FedGAN PoC"""
    print("=" * 60)
    print("FedGAN PoC - Algorithm 1 Implementation")
    print("=" * 60)
    print()
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    
    # Load and partition MNIST dataset
    print("Loading and partitioning MNIST dataset...")
    datasets, weights = load_and_partition_mnist(
        num_agents=config.NUM_AGENTS,
        digits_per_agent=config.DIGITS_PER_AGENT,
        batch_size=config.BATCH_SIZE,
        buffer_size=config.BUFFER_SIZE
    )
    
    # Initialize trainer
    trainer = FedGANTrainer(
        num_agents=config.NUM_AGENTS,
        num_epochs=config.NUM_EPOCHS,
        sync_interval=config.SYNC_INTERVAL,
        noise_dim=config.NOISE_DIM,
        learning_rate_disc=config.LEARNING_RATE_DISCRIMINATOR,
        learning_rate_gen=config.LEARNING_RATE_GENERATOR,
        output_dir=config.OUTPUT_DIR
    )
    
    # Setup agents
    trainer.setup(datasets, weights)
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("FedGAN PoC Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.OUTPUT_DIR}/")
    print(f"  - Animated GIF: {config.GIF_FILENAME}")
    print(f"  - Loss graph: {config.LOSS_PNG_FILENAME}")
    print(f"  - Loss data: {config.LOSS_CSV_FILENAME}")
    print(f"  - Per-epoch images: images/")


if __name__ == '__main__':
    main()
