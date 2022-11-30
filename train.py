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

from gan import (
  CycleGan,
  create_generator_state,
  create_discriminator_state,
  generator_step,
  discriminator_step,
)
import data as dataset
import image_pool
import options.train_options as TrainOptions
 

learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
# train.train(opts)

def train(opts={}): 
  print("Training with configuration: ")
  pprint.pprint(opts)

  default_opts = TrainOptions.TrainOptions()
  opts = {**default_opts, **opts}

  model = CycleGan()
  training_data = dataset.create_dataset()

  # Initialize States
  g_state = create_generator_state(model, opts.learning_rate, opts.beta1) #contain apply_fn=None, params of both G_A and G_B, and optimizer
  d_A_state = create_discriminator_state(model, opts.learning_rate, opts.beta1) #contain apply_fn=None, params of both D_A and D_B, and optimizer
  d_B_state = create_discriminator_state(model, opts.learning_rate, opts.beta1) #contain apply_fn=None, params of both D_A and D_B, and optimizer

  # Initialize Image Pools 
  pool_A = image_pool.ImagePool(opts.pool_size)
  pool_B = image_pool.ImagePool(opts.pool_size)

  for _ in range(opts["epoch"]):
    real_data = next(training_data)

    # G step 
    loss, g_state, generated_data = generator_step( #=========================>
      model, g_state, d_A_state, d_B_state, real_data
    )
    fake_B, _, fake_A, _ = generated_data

    # Pool
    fake_A = pool_A.query(fake_A)
    fake_B = pool_B.query(fake_B)

    # D step 
    loss, d_A_state, d_B_state = discriminator_step(
      model, d_A_state, d_B_state, real_data, (fake_A, fake_B),
    ) 
    # TODO: Fill in tuples for real and fake data



  



