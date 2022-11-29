""" Training and testing functions.

References:
    Learning rate scheduling docs:
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
	https://optax.readthedocs.io/en/latest/api.html

	Author Implementation: 
	https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/cycle_gan_model.py
"""

import itertools

from flax.training.train_state import TrainState
import flax.linen as nn
import jax.numpy as jnp
import optax

from image_pool import ImagePool
from networks import Discriminator, Generator
import config
import itertools


class CycleGan():
	# TODO: putting model options here ok? 
	pool_size: int 

	def __init__(self, rng=1337):

		self.G_A2B = Generator()
		self.G_B2A = Generator()

		self.params_G_A2B = self.G_A2B.init(rng, jnp.ones([1, 28, 28, 1]))['params']
		self.params_G_A2B = self.G_A2B.init(rng, jnp.ones([1, 28, 28, 1]))['params']
		self.optimizer_G = optax.adam(config.learning_rate, b1=config.beta1)
		
  		# Create states for all optimizers and parameters
		self.state_G = TrainState.create(
      		apply_fn=self.self.train_fn_G, params=itertools.chain(), tx=self.optimizer_G
		)

		self.D_A = Discriminator()
		self.D_B = Discriminator()
		self.optimizer_D = optax.adam(config.learning_rate, b1=config.beta1)
		self.state_D = TrainState.create(
			apply_fn=self.train_fn_D, params=itertools.chain(), tx=self.optimizer_D
		)

	def train_fn_G(self):
		

	def train_fn_G(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		self.fake_B = self.netG_A(self.real_A)  # G_A(A)
		self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
		self.fake_A = self.netG_B(self.real_B)  # G_B(B)
		self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

	def train_fn_D(self): 
		pred_real = self.D_A
	
	def setup(self):
		self.G_A2B = None # Fill in 
		self.G_B2A = None # Fill in 

		if self.training: 
			self.D_A = None # Fill in 
			self.D_B = None # Fill in 

		if self.training:
			self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
			self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
	
	def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		# Train generator
		self.forward()      # compute fake images and reconstruction images.
		

		self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
		self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

		# self.backward_G()             # calculate gradients for G_A and G_B
		# self.optimizer_G.step()       # update G_A and G_B's weights
		self.state_G.apply_fn()
		self.state_G.apply_gradients()

		# Train discriminator
		self.set_requires_grad([self.netD_A, self.netD_B], True)
		self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
		self.backward_D_A()      # calculate gradients for D_A
		self.backward_D_B()      # calculate graidents for D_B
		self.optimizer_D.step()  # update D_A and D_B's weights

	# This is the equivalent of the forward() method in PyTorch.
	def __call__(self, x):
		fake_B = self.G_A2B(self.real_A)  # G_A(A)
		rec_A = self.G_B2A(self.fake_B)   # G_B(G_A(A))
		fake_A = self.G_B2A(self.real_B)  # G_B(B)
		rec_B = self.G_A2B(self.fake_A)   # G_A(G_B(B))
		return fake_B, rec_A, fake_A, rec_B

		



