""" Training 

References: 
    https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy

    https://github.com/google/flax/blob/main/examples/imagenet/train.py
    https://github.com/google/flax/blob/main/examples/mnist/train.py
"""

import itertools

from flax.training.train_state import TrainState
import numpy as np
import jax.numpy as jnp
import jax
import optax
import pprint
from tqdm import tqdm
import torch 

from gan import (
    CycleGan,
    create_generator_state,
    create_discriminator_state,
    generator_step,
    discriminator_step,
)
import data as dataset
import image_pool

from types import SimpleNamespace


def train(model_opts, dataset_opts):
    print("Training with configuration: ")
    pprint.pprint(model_opts)

    model_opts = SimpleNamespace(**model_opts)
    model = CycleGan(model_opts)
    training_data = dataset.create_dataset(dataset_opts)


    # Initialize States
    key = jax.random.PRNGKey(1337)
    g_state = create_generator_state(
        key,
        model,
        model_opts.input_shape,
        model_opts.learning_rate,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both G_A and G_B, and optimizer
    d_A_state = create_discriminator_state(
        key,
        model,
        model_opts.input_shape,
        model_opts.learning_rate,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both D_A and D_B, and optimizer
    d_B_state = create_discriminator_state(
        key,
        model,
        model_opts.input_shape,
        model_opts.learning_rate,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both D_A and D_B, and optimizer

    # Initialize Image Pools
    pool_A = image_pool.ImagePool(model_opts.pool_size)
    pool_B = image_pool.ImagePool(model_opts.pool_size)

    epoch_g_losses = []
    epoch_d_a_losses = []
    epoch_d_b_losses = []
    for i in range(model_opts.epochs):
        g_losses = []
        d_a_losses = []
        d_b_losses = []
        print(f"\n========START OF EPOCH {i}========")
        for j, data in tqdm(enumerate(training_data)):
            real_A = data["A"]
            real_B = data["B"]
            real_A = torch.permute(real_A, (0, 2, 3, 1)).numpy()
            real_B = torch.permute(real_B, (0, 2, 3, 1)).numpy()

            real_A, real_B = jnp.array(real_A), jnp.array(real_B)
            # breakpoint()
            # G step
            loss, g_state, generated_data = generator_step(
                model, g_state, d_A_state, d_B_state, (real_A, real_B)
            )
            fake_B, _, fake_A, _ = generated_data
            g_losses.append(loss)

            # Pool
            fake_A = pool_A.query(fake_A)
            fake_B = pool_B.query(fake_B)

            # D step
            loss_A, loss_B, d_A_state, d_B_state = discriminator_step(
                model,
                d_A_state,
                d_B_state,
                (real_A, real_B),
                (fake_A, fake_B),
            )
            d_a_losses.append(loss_A)
            d_b_losses.append(loss_B)

            # TODO: Fill in tuples for real and fake data

        avg_g_loss = jnp.mean(g_losses)
        avg_d_a_loss = jnp.mean(d_a_losses)
        avg_d_b_loss = jnp.mean(d_b_losses)
        print(f"Epoch {i} Avg G Loss: {avg_g_loss}")
        print(f"Epoch {i} Avg D_A Loss: {avg_d_a_loss}")
        print(f"Epoch {i} Avg D_B Loss: {avg_d_b_loss}")
        epoch_g_losses.append(avg_g_loss)
        epoch_d_a_losses.append(avg_d_a_loss)
        epoch_d_b_losses.append(avg_d_b_loss)

    # TODO: Plot losses, visualize generated images

model_opts = {
    "input_shape": [1, 256, 256, 3],
    "output_nc": 3,
    "ngf": 32,
    "n_res_blocks": 6,
    "use_dropout": True,
    "ndf": 64,
    "netD": "n_layers",
    "n_layers": 3,
    "gan_mode": "lsgan",  # default value from github [vanilla| lsgan | wgangp]
    "epochs": 100,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "initializer": jax.nn.initializers.normal(stddev=0.02),
    "pool_size": 50,
    # @source https://github.com/junyanz/CycleGAN/issues/68
    # Lambdas are set with defaults from the source code
    # @source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/cycle_gan_model.py#L41
    "lambda_A": 10.0,
    "lambda_B": 10.0, 
    "lambda_id": 0.5, 
}

dataset_opts = {
    "dataset_mode": "unaligned",
    "max_dataset_size": float("inf"),
    "preprocess": "resize_and_crop",
    "no_flip": True,
    "display_winsize": 256,
    "num_threads": 4,
    "batch_size": 1,
    "load_size": 286,
    "crop_size": 256,
    "dataroot": "./horse2zebra",
    "phase": "train",
    "direction": "AtoB",
    "input_nc": 3,
    "output_nc": 3,
    "serial_batches": True,
}

# train.train(temp_opts)
