"""
Data loading and partitioning for FedGAN.

Implements non-IID data partitioning where each agent receives
a subset of MNIST classes (2 classes per agent for 5 agents).

Corresponds to Algorithm 1 Input: Local datasets R_i for each agent i.
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple


def load_and_prepare_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST dataset and normalize to [-1, 1] range.
    
    Returns:
        Tuple of (images, labels) where images are normalized to [-1, 1]
    """
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    
    # Reshape to (N, 28, 28, 1)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    
    # Normalize to [-1, 1] (important for tanh activation in generator)
    train_images = (train_images - 127.5) / 127.5
    
    return train_images, train_labels


def create_non_iid_partitions(
    images: np.ndarray,
    labels: np.ndarray,
    num_agents: int = 5,
    classes_per_agent: int = 2
) -> List[Tuple[tf.data.Dataset, int]]:
    """
    Partition MNIST into non-IID datasets for each agent.
    Each agent receives exactly 2 classes.
    
    Agent assignments:
    - Agent 0: classes 0, 1
    - Agent 1: classes 2, 3
    - Agent 2: classes 4, 5
    - Agent 3: classes 6, 7
    - Agent 4: classes 8, 9
    
    Args:
        images: MNIST images normalized to [-1, 1]
        labels: MNIST labels (0-9)
        num_agents: Number of federated agents (default: 5)
        classes_per_agent: Number of classes per agent (default: 2)
    
    Returns:
        List of (dataset, dataset_size) tuples, one per agent
        Corresponds to R_i in Algorithm 1
    """
    agent_datasets = []
    
    for agent_id in range(num_agents):
        # Determine which classes this agent should have
        start_class = agent_id * classes_per_agent
        end_class = start_class + classes_per_agent
        agent_classes = list(range(start_class, end_class))
        
        # Filter data for this agent's classes
        mask = np.isin(labels, agent_classes)
        agent_images = images[mask]
        agent_labels = labels[mask]
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((agent_images, agent_labels))
        dataset_size = len(agent_images)
        
        agent_datasets.append((dataset, dataset_size))
        
        print(f"Agent {agent_id}: classes {agent_classes}, {dataset_size} samples")
    
    return agent_datasets


def calculate_agent_weights(dataset_sizes: List[int]) -> np.ndarray:
    """
    Calculate aggregation weights p_i for each agent based on dataset size.
    
    According to Algorithm 1 Step 3:
    p_i = |R_i| / sum_j |R_j|
    
    Args:
        dataset_sizes: List of dataset sizes for each agent
    
    Returns:
        Array of weights that sum to 1.0
    """
    total_samples = sum(dataset_sizes)
    weights = np.array([size / total_samples for size in dataset_sizes])
    
    print(f"\nAgent weights (p_i): {weights}")
    print(f"Sum of weights: {weights.sum():.6f}")
    
    return weights


def prepare_federated_data(
    batch_size: int = 64,
    num_agents: int = 5
) -> Tuple[List[tf.data.Dataset], np.ndarray]:
    """
    Prepare complete federated dataset setup.
    
    Returns:
        Tuple of:
        - List of batched datasets (one per agent)
        - Array of aggregation weights p_i
    """
    # Load MNIST
    images, labels = load_and_prepare_mnist()
    
    # Create non-IID partitions
    agent_data = create_non_iid_partitions(images, labels, num_agents)
    
    # Extract datasets and sizes
    datasets = []
    sizes = []
    for dataset, size in agent_data:
        # Shuffle and batch
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
        datasets.append(dataset)
        sizes.append(size)
    
    # Calculate aggregation weights
    weights = calculate_agent_weights(sizes)
    
    return datasets, weights


if __name__ == "__main__":
    # Test data partitioning
    print("Testing FedGAN data partitioning...")
    print("=" * 60)
    
    datasets, weights = prepare_federated_data(batch_size=64, num_agents=5)
    
    print("\n" + "=" * 60)
    print("Data partitioning test complete!")
    print(f"Created {len(datasets)} agent datasets")
    print(f"Aggregation weights: {weights}")
