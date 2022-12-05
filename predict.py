import os

from flax.training import checkpoints
from types import SimpleNamespace
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import logger

from gan import (
    CycleGan,
    create_generator_state,
    generator_prediction,
)
from train import create_lr_schedule_fn
from img_utils import array_to_img, img_to_array


def predict(model_opts, start: str, save_img=True, plt_img=False):
    model_opts = SimpleNamespace(**model_opts)
    filename = model_opts.data_path
    end = "B" if start == "A" else "A"
    real_data = jnp.expand_dims(img_to_array(filename), axis=0)

    # Restore states
    logger.info("Restoring states...")
    key = jax.random.PRNGKey(1337)
    model = CycleGan(model_opts)
    key, g_state = create_generator_state(
        key,
        model,
        model_opts.input_shape,
        create_lr_schedule_fn(model_opts, 1),
        model_opts.beta1,
    )  # contain apply_fn=None, params of both G_A and G_B, and optimizer
    g_state = checkpoints.restore_checkpoint(
        model_opts.checkpoint_directory_G, target=g_state
    )

    logger.info(f"Generating {model_opts.model_name} prediction, {start} to {end}")
    key, generated_data = generator_prediction(key, model, g_state, real_data, start)

    fake, recover = generated_data

    # Write latest generated images from validation set to disk
    if save_img:
        array_to_img(
            fake[0],
            os.path.join(
                model_opts.pred_img_path,
                f"fake_{end}_{filename.split('/')[-1][:-4]}.jpg",
            ),
        )
        array_to_img(
            recover[0],
            os.path.join(
                model_opts.pred_img_path,
                f"recover_{start}_{filename.split('/')[-1][:-4]}.jpg",
            ),
        )

    # Plot latest generated images from validation set in Jupyter notebook
    if plt_img:
        fig, ax = plt.subplots(1, 2)
        ax[0, 0] = ax.imshow(fake[0])
        ax[0, 0].title.set_text(f"Fake {end}")
        ax[0, 1] = ax.imshow(recover[0])
        ax[0, 1].title.set_text(f"Recover {start}")
