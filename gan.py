""" Training and testing functions.

References:
    Learning rate scheduling docs:
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
	https://optax.readthedocs.io/en/latest/api.html
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from image_pool import ImagePool
import networks
import optax
import itertools

class CycleGan(nn.Module):
	# TODO: putting model options here ok? 
	training: bool 
	pool_size: int 
	learning_rate: float 
	beta1: float
	
	def setup(self):

		self.G_A2B = None
		self.G_B2A = None

		if self.training: 
			self.D_A = None
			self.D_B = None

		# if self.training:
		self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
		self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
		# define loss functions
		self.criterionGAN = networks.GANLoss().to(self.device)  # define GAN loss.
		self.criterionCycle = networks.L1Loss()
		self.criterionIdt = networks.L1Loss()
		# initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
		self.optimizer_G = optax.adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.learning_rate, betas=(self.beta1, 0.999))
		self.optimizer_D = optax.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
		self.optimizers.append(self.optimizer_G)
		self.optimizers.append(self.optimizer_D)
	
	def __call__(self, x):
		pass

	