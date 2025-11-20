"""
Generator and Discriminator models for FedGAN.

Implements ACGAN-style architecture as described in the paper's appendix.
Each agent has its own local Generator and Discriminator.

Architecture follows paper specifications for MNIST:
- Generator: 100-dim noise -> 1024 -> 128 -> Conv4x4 -> Conv4x4 -> 28x28x1
- Discriminator: 28x28x1 -> Conv4x4 -> Conv4x4 -> 1024 -> binary output
"""

import tensorflow as tf
from tensorflow.keras import layers


def make_generator(noise_dim: int = 100) -> tf.keras.Model:
    """
    Create Generator model following paper's ACGAN architecture.
    
    Architecture (from paper appendix):
    - Linear: noise_dim -> 1024, BatchNorm, ReLU
    - Linear: 1024 -> 128 * 7 * 7, BatchNorm, ReLU
    - Reshape to (7, 7, 128)
    - TransposedConv: 4x4 kernel, stride 2, 64 filters, BatchNorm, ReLU
    - TransposedConv: 4x4 kernel, stride 2, 1 filter, Tanh
    - Output: 28x28x1
    
    Corresponds to theta parameters in Algorithm 1.
    
    Args:
        noise_dim: Dimension of input noise vector (default: 100)
    
    Returns:
        Generator model
    """
    model = tf.keras.Sequential(name='generator')
    
    # Linear layer: noise_dim -> 1024
    model.add(layers.Dense(1024, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Linear layer: 1024 -> 128 * 7 * 7
    model.add(layers.Dense(128 * 7 * 7, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Reshape to 3D tensor for convolutions
    model.add(layers.Reshape((7, 7, 128)))
    assert model.output_shape == (None, 7, 7, 128)
    
    # Transposed Conv: 4x4 kernel, stride 2, 64 filters -> 14x14x64
    model.add(layers.Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same', use_bias=False
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    assert model.output_shape == (None, 14, 14, 64)
    
    # Transposed Conv: 4x4 kernel, stride 2, 1 filter -> 28x28x1
    model.add(layers.Conv2DTranspose(
        1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'
    ))
    assert model.output_shape == (None, 28, 28, 1)
    
    return model


def make_discriminator() -> tf.keras.Model:
    """
    Create Discriminator model following paper's ACGAN architecture.
    
    Architecture (from paper appendix):
    - Conv: 4x4 kernel, stride 2, 64 filters, LeakyReLU(0.2)
    - Conv: 4x4 kernel, stride 2, 128 filters, BatchNorm, LeakyReLU(0.2)
    - Flatten
    - Linear: -> 1024, BatchNorm, LeakyReLU(0.2)
    - Linear: -> 1 (binary classification)
    - Output: logits (no sigmoid, use from_logits=True in loss)
    
    Corresponds to w parameters in Algorithm 1.
    
    Returns:
        Discriminator model
    """
    model = tf.keras.Sequential(name='discriminator')
    
    # Conv: 4x4 kernel, stride 2, 64 filters -> 14x14x64
    model.add(layers.Conv2D(
        64, (4, 4), strides=(2, 2), padding='same',
        input_shape=[28, 28, 1]
    ))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 14, 14, 64)
    
    # Conv: 4x4 kernel, stride 2, 128 filters -> 7x7x128
    model.add(layers.Conv2D(
        128, (4, 4), strides=(2, 2), padding='same', use_bias=False
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 7, 7, 128)
    
    # Flatten
    model.add(layers.Flatten())
    
    # Linear: -> 1024
    model.add(layers.Dense(1024, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Linear: -> 1 (binary classification, output logits)
    model.add(layers.Dense(1))
    
    return model


def create_agent_models(num_agents: int = 5, noise_dim: int = 100):
    """
    Create Generator and Discriminator pairs for all agents.
    
    According to Algorithm 1 Input:
    Initialize w_i_0 = w_hat and theta_i_0 = theta_hat for all agents.
    
    Args:
        num_agents: Number of federated agents (default: 5)
        noise_dim: Dimension of generator input noise (default: 100)
    
    Returns:
        Tuple of (generators_list, discriminators_list)
    """
    generators = []
    discriminators = []
    
    for i in range(num_agents):
        gen = make_generator(noise_dim)
        disc = make_discriminator()
        
        generators.append(gen)
        discriminators.append(disc)
        
        print(f"Agent {i}: Created Generator and Discriminator")
    
    return generators, discriminators


# Loss functions for GAN training
def discriminator_loss(real_output, fake_output):
    """
    Binary cross-entropy loss for discriminator.
    
    Corresponds to g_tilde_i gradient computation in Algorithm 1 Step 1.
    
    Args:
        real_output: Discriminator output on real images
        fake_output: Discriminator output on fake images
    
    Returns:
        Total discriminator loss
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Real images should be classified as 1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    
    # Fake images should be classified as 0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    Binary cross-entropy loss for generator.
    
    Corresponds to h_tilde_i gradient computation in Algorithm 1 Step 1.
    
    Args:
        fake_output: Discriminator output on generated images
    
    Returns:
        Generator loss
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Generator wants discriminator to classify fake images as real (1)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


if __name__ == "__main__":
    # Test model creation
    print("Testing FedGAN model creation...")
    print("=" * 60)
    
    # Create single generator and discriminator for testing
    gen = make_generator(noise_dim=100)
    disc = make_discriminator()
    
    print("\nGenerator architecture:")
    gen.summary()
    
    print("\nDiscriminator architecture:")
    disc.summary()
    
    # Test forward pass
    noise = tf.random.normal([1, 100])
    generated_image = gen(noise, training=False)
    print(f"\nGenerated image shape: {generated_image.shape}")
    
    decision = disc(generated_image, training=False)
    print(f"Discriminator output shape: {decision.shape}")
    
    print("\n" + "=" * 60)
    print("Model creation test complete!")
