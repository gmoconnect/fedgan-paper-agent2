"""
Quick test of FedGAN training - runs for just 2 epochs to verify implementation
"""
import sys
sys.path.insert(0, '.')

# Override EPOCHS to 2 for testing
import train
train.EPOCHS = 2

if __name__ == "__main__":
    print("Running FedGAN test with 2 epochs...")
    train.train_fedgan(epochs=2)
    print("\nTest complete! Check outputs/ directory for results.")
