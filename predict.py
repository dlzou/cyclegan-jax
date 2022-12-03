from flax.training import checkpoints
from types import SimpleNamespace
from tqdm import tqdm
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import logger

from gan import (
    CycleGan,
    create_generator_state,
    generator_prediction,
)
from img_utils import array_to_img, img_to_array


def predict(model_opts, filename: str, save_img=True, plt_img=False):
    if "trainA" in filename or "testA" in filename:
        direction = "A"
    elif "trainB" in filename or "testB" in filename:
        direction = "B"
    real_data = jnp.expand_dims(img_to_array(filename), axis=0)

    # Restore states
    logger.info("Restoring states...")
    model_opts = SimpleNamespace(**model_opts)
    key = jax.random.PRNGKey(1337)
    model = CycleGan(model_opts)
    key, g_state = create_generator_state(
        key,
        model,
        model_opts.input_shape,
        model_opts.learning_rate,
        model_opts.beta1,
    )  # contain apply_fn=None, params of both G_A and G_B, and optimizer
    g_state = checkpoints.restore_checkpoint(
        model_opts.checkpoint_directory_G, target=g_state
    )

    key, generated_data = generator_prediction(
        key, model, g_state, real_data, direction
    )

    fake, recover = generated_data

    # Write latest generated images from validation set to disk
    result = "B" if direction == "A" else "A"
    if save_img:
        array_to_img(
            fake[0], f"pred_img/fake_{result}_{filename.split('/')[-1][:-4]}.jpg"
        )
        array_to_img(
            recover[0], f"pred_img/fake_{direction}_{filename.split('/')[-1][:-4]}.jpg"
        )

    # Plot latest generated images from validation set in Jupyter notebook
    if plt_img:
        fig, ax = plt.subplots(1, 2)
        ax[0, 0] = ax.imshow(fake[0])
        ax[0, 0].title.set_text(f"Fake {result}")
        ax[0, 1] = ax.imshow(recover[0])
        ax[0, 1].title.set_text(f"Recover {direction}")
