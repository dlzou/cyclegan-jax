""" Training 

References: 
    https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy

    https://github.com/google/flax/blob/main/examples/imagenet/train.py
    https://github.com/google/flax/blob/main/examples/mnist/train.py
"""

from types import SimpleNamespace
import os

from flax.training import checkpoints
from tqdm import tqdm
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch

from gan import (
    CycleGan,
    create_generator_state,
    create_discriminator_state,
    generator_step,
    generator_validation,
    discriminator_step,
)
import dataset
import image_pool
from img_utils import array_to_img
import logger


def create_lr_schedule_fn(opts, steps_per_epoch):
    """
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
    Args:
        decay_after_epochs: specify epoch after which lr linearly decays to 0.
        base_lr: initial learning rate
        stepcs_per_epoch: steps per epoch
    """
    schedule_fn = optax.piecewise_interpolate_schedule(
        interpolate_type="linear",
        init_value=opts.learning_rate,
        boundaries_and_scales={
            opts.decay_after_epochs * steps_per_epoch: 1.0,
            opts.epochs * steps_per_epoch: 0.0,
        },
    )
    return schedule_fn


def train(model_opts, dataset_opts, save_img=True, plt_img=False):
    model_opts = SimpleNamespace(**model_opts)
    training_data, validation_data = dataset.create_dataset(dataset_opts)

    logger.info("Cleaning and creating val_img/ directory ...")
    os.system(f"rm -f {os.path.join(model_opts.val_img_path, '*')}")
    os.system(f"mkdir -p {model_opts.val_img_path}")
    logger.info(f"Training with configuration: {model_opts}")

    # Create learning rate scheduler
    lr_schedule_fn = create_lr_schedule_fn(model_opts, len(training_data))

    # Initialize States
    key = jax.random.PRNGKey(1337)
    model = CycleGan(model_opts)

    logger.info("Creating CycleGAN states and initializing the image pool...")
    key, g_state = create_generator_state(
        key,
        model,
        model_opts.input_shape,
        lr_schedule_fn,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both G_A and G_B, and optimizer
    key, d_A_state = create_discriminator_state(
        key,
        model,
        model_opts.input_shape,
        lr_schedule_fn,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both D_A and D_B, and optimizer
    key, d_B_state = create_discriminator_state(
        key,
        model,
        model_opts.input_shape,
        lr_schedule_fn,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both D_A and D_B, and optimizer

    # Initialize image pools
    pool_A = image_pool.ImagePool(model_opts.pool_size)
    pool_B = image_pool.ImagePool(model_opts.pool_size)

    epoch_g_train_losses = []
    epoch_d_a_train_losses = []
    epoch_d_b_train_losses = []

    epoch_g_val_losses = []

    learning_rates = []

    # for epoch in range(model_opts.epochs):
    for epoch in range(model_opts.epochs):
        logger.info(
            f"\n========{model_opts.model_name}: Start of Epoch {epoch}========"
        )
        lr = lr_schedule_fn(g_state.step)
        logger.info(f"Epoch {epoch} learning rate: {lr}")
        learning_rates.append(lr)
        logger.info(
            f"g_state.step={g_state.step}, d_a_state.step={d_A_state.step}, d_b_state.step={d_A_state.step}"
        )

        # Training stage
        g_train_losses = []
        d_a_train_losses = []
        d_b_train_losses = []
        for j, data in tqdm(enumerate(training_data)):
            real_A = data["A"]
            real_A = torch.permute(real_A, (0, 2, 3, 1)).numpy()
            real_B = data["B"]
            real_B = torch.permute(real_B, (0, 2, 3, 1)).numpy()

            # G step
            key, loss, g_state, generated_data = generator_step(
                key, model, g_state, d_A_state, d_B_state, (real_A, real_B)
            )
            fake_B, _, fake_A, _ = generated_data
            g_train_losses.append(loss)

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
            d_a_train_losses.append(loss_A)
            d_b_train_losses.append(loss_B)

        avg_g_train_loss = jnp.mean(jnp.array(g_train_losses))
        avg_d_a_train_loss = jnp.mean(jnp.array(d_a_train_losses))
        avg_d_b_train_loss = jnp.mean(jnp.array(d_b_train_losses))
        epoch_g_train_losses.append(avg_g_train_loss)
        epoch_d_a_train_losses.append(avg_d_a_train_loss)
        epoch_d_b_train_losses.append(avg_d_b_train_loss)
        logger.info(f"Epoch {epoch} avg G training loss: {avg_g_train_loss}")
        logger.info(f"Epoch {epoch} avg D_A training loss: {avg_d_a_train_loss}")
        logger.info(f"Epoch {epoch} avg D_B training loss: {avg_d_b_train_loss}")

        logger.info("Running validation...")
        # Validation stage
        g_val_losses = []
        # TODO: create validation_data set
        for j, data in tqdm(enumerate(validation_data)):
            real_A = data["A"]
            real_B = data["B"]
            real_A = torch.permute(real_A, (0, 2, 3, 1)).numpy()
            real_B = torch.permute(real_B, (0, 2, 3, 1)).numpy()
            key, g_val_loss, generated_data = generator_validation(
                key,
                model,
                g_state,
                d_A_state,
                d_B_state,
                (real_A, real_B),
            )
            g_val_losses.append(g_val_loss)

        fake_B, _, fake_A, _ = generated_data
        logger.info("Outputting the generated image from validation...")

        # Write latest generated images from validation set to disk
        if save_img:
            A_label = data["A_paths"]
            B_label = data["B_paths"]
            for i in np.arange(fake_A.shape[0]):
                array_to_img(
                    fake_A[i],
                    os.path.join(
                        model_opts.val_img_path,
                        f"{epoch}_fake_A_{B_label[i].split('/')[-1][:-4]}.jpg",
                    ),
                )
            for i in np.arange(fake_B.shape[0]):
                array_to_img(
                    fake_B[i],
                    os.path.join(
                        model_opts.val_img_path,
                        f"{epoch}_fake_B_{A_label[i].split('/')[-1][:-4]}.jpg",
                    ),
                )

            avg_g_val_loss = jnp.mean(jnp.array(g_val_losses))
            logger.info(f"Epoch {epoch} avg G validation loss: {avg_g_val_loss}")
            epoch_g_val_losses.append(avg_g_val_loss)

        # Plot latest generated images from validation set in Jupyter notebook
        if plt_img:
            fig, ax = plt.subplots(1, 2)
            ax[0, 0] = ax.imshow(fake_B[0])
            ax[0, 0].title.set_text(f"Epoch {epoch} A to B")
            ax[0, 1] = ax.imshow(fake_A[0])
            ax[0, 1].title.set_text(f"Epoch {epoch} B to A")

        # Checkpoint the state
        # @source: https://github.com/google/flax/discussions/1876
        logger.info("Saving checkpoint...")
        g_state_checkpoint = checkpoints.save_checkpoint(
            ckpt_dir=model_opts.checkpoint_directory_G,
            target=g_state,
            step=epoch,
            overwrite=True,
        )
        logger.info(f"G state checkpoint saved at {g_state_checkpoint}")
        d_A_state_checkpoint = checkpoints.save_checkpoint(
            ckpt_dir=model_opts.checkpoint_directory_D_A,
            target=d_A_state,
            step=epoch,
            overwrite=True,
        )
        logger.info(f"D_A state checkpoint saved at {d_A_state_checkpoint}")
        d_B_state_checkpoint = checkpoints.save_checkpoint(
            ckpt_dir=model_opts.checkpoint_directory_D_B,
            target=d_B_state,
            step=epoch,
            overwrite=True,
        )
        logger.info(f"D_B state checkpoint saved at {d_B_state_checkpoint}")

    return (
        epoch_g_train_losses,
        epoch_d_a_train_losses,
        epoch_d_b_train_losses,
        epoch_g_val_losses,
        learning_rates,
    )


def get_default_opts(data_path, model_path):
    model_opts = {
        "input_shape": [1, 256, 256, 3],
        "output_nc": 3,
        "ngf": 32,
        "n_res_blocks": 6,
        "dropout_rate": 0.5,
        "ndf": 64,
        "netD": "n_layers",
        "n_layers": 3,
        "gan_mode": "wgangp",  # [vanilla | lsgan | wgangp]
        "epochs": 150,
        "decay_after_epochs": 75,
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
        "data_path": data_path,
        "model_path": model_path,
        "model_name": model_path.split("/")[-1],
        "checkpoint_directory_G": os.path.join(
            model_path, "model_checkpoints", "checkpoint_G"
        ),
        "checkpoint_directory_D_A": os.path.join(
            model_path, "model_checkpoints", "checkpoint_D_A"
        ),
        "checkpoint_directory_D_B": os.path.join(
            model_path, "model_checkpoints", "checkpoint_D_B"
        ),
        "val_img_path": os.path.join(model_path, "val_img"),
        "pred_img_path": os.path.join(model_path, "pred_img"),
    }

    dataset_opts = {
        "dataset_mode": "unaligned",
        "max_dataset_size": float("inf"),
        "preprocess": "resize_and_crop",
        "no_flip": True,
        "display_winsize": 256,
        "num_threads": 4,
        "train_set_ratio": 0.85,
        "batch_size": 1,
        "load_size": 286,
        "crop_size": 256,
        "dataroot": f"./{data_path}",
        "direction": "AtoB",
        "input_nc": 3,
        "output_nc": 3,
        "serial_batches": False,
    }
    return model_opts, dataset_opts
