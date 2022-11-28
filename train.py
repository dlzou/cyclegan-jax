""" Training 

References: 
	https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy

	https://github.com/google/flax/blob/main/examples/imagenet/train.py
	https://github.com/google/flax/blob/main/examples/mnist/train.py
"""

from gan import CycleGan
import jax 
import jax.numpy as jnp
from flax.training import train_state
import optax
import torch 
import networks
import itertools
import datasets
import numpy as np


cycleGan = CycleGan([12, 8, 4])

# ----------------- Training Configurations ----------------- #
batch_size = 1 # Placeholder
learning_rate = 0.001 # Placeholder
beta1 = 0.5 # Placeholder
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = jnp.ones(shape=(batch_size, 32, 32, 3)) # Dummy Input

# Do we need to keep track of 4 states since we have 4 models?
def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  model = cycleGan()
  params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # TODO: Change 
  optimizer_G = optax.adam(itertools.chain(model.netG_A.parameters(), model.netG_B.parameters()), lr=learning_rate, betas=(beta1, 0.999))
  optimizer_D = optax.adam(itertools.chain(model.netD_A.parameters(), model.netD_B.parameters()), lr=learning_rate, betas=(beta1, 0.999))
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=[optimizer_G, optimizer_D])

@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  # define loss functions
  criterionGAN = networks.GanLoss().to(device)  # define GAN loss.
  criterionCycle = networks.L1Loss().to(device)  # define cycle loss.
  criterionIdt = networks.L1Loss().to(device)  # define identity loss.

	# TODO: Finish applying model 
	# forward
	# compute fake images and reconstruction images.
	# G_A and G_B
	# Ds require no gradients when optimizing Gs
	# set G_A and G_B's gradients to zero
	# calculate gradients for G_A and G_B
	# update G_A and G_B's weights
	# D_A and D_B
	# set D_A and D_B's gradients to zero
	# calculate gradients for D_A
	# calculate graidents for D_B
	# update D_A and D_B's weights

  grad_fn = jax.value_and_grad(criterionGAN, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def train_and_evaluate(config, workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, test_ds = datasets.get_datasets()
  rng = jax.random.PRNGKey(0)


  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    config.batch_size,
                                                    input_rng)
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'])
	# TODO: Training logs 

  return state









# def train_model(
#     model, criterion, optimizer, num_epochs, train_ds, val_ds
# ):
#     for epoch in range(num_epochs):
#         running_train_loss, running_val_loss = 0., 0.
#         # ============== Training ============== #
#         model.train()
#         for inputs, labels in train_ds:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward
#             # track history if only in train
#             with torch.set_grad_enabled(True):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#                 # backward + optimize
#                 loss.backward()
#                 optimizer.step()
#             running_train_loss += loss.item() * inputs.size(0)

# 			# forward
#         cycleGan.forward()      # compute fake images and reconstruction images.
#         # G_A and G_B
#         cycleGan.set_requires_grad([cycleGan.netD_A, cycleGan.netD_B], False)  # Ds require no gradients when optimizing Gs
#         cycleGan.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
#         cycleGan.backward_G()             # calculate gradients for G_A and G_B
#         cycleGan.optimizer_G.step()       # update G_A and G_B's weights
#         # D_A and D_B
#         cycleGan.set_requires_grad([cycleGan.netD_A, cycleGan.netD_B], True)
#         cycleGan.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
#         cycleGan.backward_D_A()      # calculate gradients for D_A
#         cycleGan.backward_D_B()      # calculate graidents for D_B
#         cycleGan.optimizer_D.step()  # update D_A and D_B's weights

