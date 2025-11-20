"""Configuration parameters for FedGAN PoC

Maps to Algorithm 1 parameters from the paper.
"""

# Algorithm 1: Training period N (in epochs)
NUM_EPOCHS = 50

# Algorithm 1: Number of agents B
NUM_AGENTS = 5

# Algorithm 1: Synchronization interval K (in iterations/batches)
SYNC_INTERVAL = 20

# Algorithm 1: Learning rates a(n) and b(n)
# Using equal time-scale: a(n) = b(n)
LEARNING_RATE_DISCRIMINATOR = 1e-4  # a(n)
LEARNING_RATE_GENERATOR = 1e-4      # b(n)

# Dataset and training parameters
BATCH_SIZE = 256
BUFFER_SIZE = 60000
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# Data partitioning (non-iid)
# Each agent gets 2 digit classes
DIGITS_PER_AGENT = 2

# Output directories
OUTPUT_DIR = 'outputs'
IMAGES_DIR = 'outputs/images'
CHECKPOINTS_DIR = 'outputs/checkpoints'

# Visualization
GIF_FILENAME = 'training_progress.gif'
GIF_DURATION = 0.5  # seconds per frame
LOSS_PNG_FILENAME = 'losses.png'
LOSS_CSV_FILENAME = 'losses.csv'

# Random seed for reproducibility
SEED = 42
