#!/usr/bin/env python3
"""DCGAN training CLI based on TensorFlow tutorial

Implements model, loss, optimizer as in TensorFlow DCGAN tutorial.
Saves per-epoch generated images, creates an animated GIF showing progress,
and writes loss CSV and PNG combining generator and discriminator losses.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def make_generator_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'
        )
    )
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model() -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def prepare_dataset(dataset_name: str, buffer_size: int, batch_size: int) -> tf.data.Dataset:
    if dataset_name.lower() == 'mnist':
        (train_images, _), _ = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        # Normalize to [-1, 1]
        train_images = (train_images - 127.5) / 127.5
        dataset = tf.data.Dataset.from_tensor_slices(train_images)
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        return dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def generate_and_save_images(model: tf.keras.Model, epoch: int, test_input: tf.Tensor, out_dir: str) -> str:
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
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
    parser = argparse.ArgumentParser(description='Train a DCGAN (TensorFlow tutorial)')
    parser.add_argument('--dataset', default='mnist', help='Dataset to use (mnist)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=60000, help='Shuffle buffer size')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of generator noise vector')
    parser.add_argument('--num_examples_to_generate', type=int, default=16, help='Number of images to generate per epoch')
    parser.add_argument('--output_dir', default='training_outputs', help='Directory to save outputs')
    parser.add_argument('--save_gif', action='store_true', help='Create an animated GIF of progress')
    parser.add_argument('--gif_duration', type=float, default=0.5, help='Frame duration in seconds for GIF')
    parser.add_argument('--seed', type=int, default=42, help='Seed for visualization (fixed)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizers')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup output directories
    out_dir = args.output_dir
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Prepare data
    train_dataset = prepare_dataset(args.dataset, args.buffer_size, args.batch_size)

    # Models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Optimizers (as tutorial)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5)

    # Checkpoints (optional)
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Seed for visualization (fixed)
    np.random.seed(args.seed)
    fixed_seed = np.random.normal(size=(args.num_examples_to_generate, args.noise_dim)).astype(np.float32)
    fixed_seed_tf = tf.convert_to_tensor(fixed_seed)

    # Loss history
    loss_history = []

    # Training loop variables
    @tf.function
    def train_step(images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        noise = tf.random.normal([tf.shape(images)[0], args.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    start_time = time.time()
    image_paths = []
    for epoch in range(1, args.epochs + 1):
        print(f'Starting epoch {epoch}/{args.epochs}')
        # accumulate per-batch losses to compute epoch averages
        epoch_gen_losses = []
        epoch_disc_losses = []
        for image_batch in train_dataset:
            # training seed for batches should be random (we use tf.random in train_step)
            g_loss, d_loss = train_step(image_batch)
            epoch_gen_losses.append(float(g_loss.numpy()))
            epoch_disc_losses.append(float(d_loss.numpy()))

        # Compute epoch averages (guard against empty)
        if epoch_gen_losses:
            avg_g_loss = sum(epoch_gen_losses) / len(epoch_gen_losses)
        else:
            avg_g_loss = 0.0
        if epoch_disc_losses:
            avg_d_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)
        else:
            avg_d_loss = 0.0

        # Save per-epoch image using fixed visualization seed
        img_path = generate_and_save_images(generator, epoch, fixed_seed_tf, images_dir)
        image_paths.append(img_path)

        print(f'Epoch {epoch} avg_gen_loss={avg_g_loss:.4f} avg_disc_loss={avg_d_loss:.4f}')
        loss_history.append({'epoch': epoch, 'gen_loss': avg_g_loss, 'disc_loss': avg_d_loss})

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    total_time = time.time() - start_time
    print(f'Training complete in {total_time:.2f}s')

    # Save losses
    loss_df = pd.DataFrame(loss_history)
    losses_png = os.path.join(out_dir, 'losses.png')
    losses_csv = os.path.join(out_dir, 'losses.csv')
    plot_and_save_losses(loss_df, losses_png, losses_csv)

    # Make GIF if requested
    if args.save_gif:
        gif_path = os.path.join(out_dir, 'training_progress.gif')
        make_gif(image_paths, gif_path, duration=args.gif_duration)
        print(f'GIF saved to {gif_path}')


if __name__ == '__main__':
    main()
