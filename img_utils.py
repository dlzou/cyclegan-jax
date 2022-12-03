import os

import jax.numpy as jnp
import numpy as np
from PIL import Image as im
import tensorflow as tf


def array_to_img(x: jnp.ndarray, filename: str) -> None:
    xnp = np.array(x)
    tf.keras.preprocessing.image.array_to_img(xnp, scale=True).save(filename)


def img_to_array(filename: str) -> jnp.ndarray:
    img = im.open(os.path.join(os.getcwd(), filename))
    ret = tf.keras.preprocessing.image.img_to_array(img) / 128 - 1
    return jnp.array(ret)


if __name__ == "__main__":
    array_to_img(
        img_to_array(
            "/home/dlzou/projects/cs182-project/horse2zebra/trainA/n02381460_8435.jpg"
        ),
        "999.jpg",
    )
