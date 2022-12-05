import os

import jax.numpy as jnp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def array_to_img(x: jnp.ndarray, filename: str) -> None:
    """
    x must have shape (H, W, C)
    """
    # xnp = np.array(x)
    # tf.keras.preprocessing.image.array_to_img(xnp, scale=True).save(filename)
    xnp = (np.array(x) + 1) * 127.5
    transforms.ToPILImage()(xnp.astype(np.uint8)).save(filename)


def img_to_array(filename: str) -> jnp.ndarray:
    img = Image.open(os.path.join(os.getcwd(), filename))
    # ret = tf.keras.preprocessing.image.img_to_array(img) / 128 - 1
    # return jnp.array(ret)
    tensor = transforms.ToTensor()(img)
    return jnp.transpose((tensor.numpy() - 0.5) * 2, (1, 2, 0))


if __name__ == "__main__":
    array_to_img(
        img_to_array(
            "/home/dlzou/projects/cs182-project/horse2zebra/trainA/n02381460_8435.jpg"
        ),
        "test.jpg",
    )
