from .animal import ANIMAL
import jax.numpy as jnp

# Note: this code is from https://github.com/google/flax/blob/acb4ce0564b76f735799d19f516c1106fee9d3cb/examples/mnist/train.py#L98
# Decide if we want to follow this convention or change 
def get_datasets():
  """Load MNIST train and test datasets into memory."""
  train_ds = None 
  test_ds = None 
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


