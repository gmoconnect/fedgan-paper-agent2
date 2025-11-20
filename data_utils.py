"""Dataset utilities for FedGAN PoC

Handles MNIST loading and non-iid partitioning across agents.
"""
from typing import List, Tuple
import numpy as np
import tensorflow as tf


def load_and_partition_mnist(
    num_agents: int,
    digits_per_agent: int,
    batch_size: int,
    buffer_size: int
) -> Tuple[List[tf.data.Dataset], List[float]]:
    """Load MNIST and partition into non-iid subsets for each agent
    
    Algorithm 1: Each agent i has local dataset R_i
    
    Args:
        num_agents: Number of agents (B in Algorithm 1)
        digits_per_agent: Number of digit classes per agent (for non-iid split)
        batch_size: Batch size for training
        buffer_size: Shuffle buffer size
        
    Returns:
        datasets: List of tf.data.Dataset, one per agent
        weights: List of agent weights p_j (data size ratios) for averaging
    """
    # Load MNIST
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    
    # Partition by digit classes (non-iid)
    # Each agent gets digits_per_agent consecutive digit classes
    agent_datasets = []
    agent_sizes = []
    
    for agent_id in range(num_agents):
        # Determine which digit classes this agent gets
        start_digit = (agent_id * digits_per_agent) % 10
        agent_digits = [(start_digit + i) % 10 for i in range(digits_per_agent)]
        
        # Filter data for this agent's digits
        mask = np.isin(train_labels, agent_digits)
        agent_images = train_images[mask]
        agent_labels = train_labels[mask]
        
        agent_sizes.append(len(agent_images))
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(agent_images)
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        agent_datasets.append(dataset)
        
        print(f"Agent {agent_id}: {len(agent_images)} samples, digits {agent_digits}")
    
    # Calculate agent weights p_j (Algorithm 1, Equation 2)
    total_samples = sum(agent_sizes)
    weights = [size / total_samples for size in agent_sizes]
    
    print(f"\nAgent weights (p_j): {weights}")
    print(f"Total samples: {total_samples}\n")
    
    return agent_datasets, weights
