import jax.numpy as jnp
import numpy as np 
from PIL import Image as im
import tensorflow as tf 
# tf.keras.utils.array_to_img


def array_to_img(x: jnp.ndarray, filename: str) -> None:
    # x_np = np.ascontiguousarray(x)
    # x_np = np.floor((x_np + 1) * 128)
    # data = im.fromarray(x_np, mode="RGB")
    # data.save(f"output_img/{filename}")
    # jax array to numpy array 
    # x_np = np.array(x)
    # # (-1, 1) -> (0, 256)
    # x_np = (x_np + 1) * 128
    # print(x_np.astype(int)[:10,0,:])
    # img = im.fromarray(x_np, 'RGB')
	xnp = np.array(x)
	tf.keras.preprocessing.image.array_to_img(xnp, scale=True).save(f"output_img/{filename}")

def img_to_array(filename: str) -> jnp.ndarray:
    # img = im.open(filename)
    # np_img_array = np.array(img)

    # # Convert to jnp 
    # img_array = jnp.array(np_img_array)
    # print(img_array[:10, 0,:])

    # # 0 - 256 to -1 - 1
    # img_array = (img_array / 128) - 1

    # return img_array
	img = im.open(filename)
	ret = tf.keras.preprocessing.image.img_to_array(img) / 128 - 1
	return jnp.array(ret)

array_to_img(img_to_array("/home/dlzou/projects/cs182-project/horse2zebra/trainA/n02381460_8435.jpg"), "999.jpg")