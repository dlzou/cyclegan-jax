""" Training and testing functions.

References:
    Learning rate scheduling docs:
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
	https://optax.readthedocs.io/en/latest/api.html

	Author Implementation: 
	https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/cycle_gan_model.py
"""

import flax.linen as nn
from image_pool import ImagePool
import optax

class CycleGan(nn.Module):
	# TODO: putting model options here ok? 
	pool_size: int 
	
	def setup(self):
		self.G_A2B = None # Fill in 
		self.G_B2A = None # Fill in 

		if self.training: 
			self.D_A = None # Fill in 
			self.D_B = None # Fill in 

		if self.training:
			self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
			self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
	
	# This is the equivalent of the forward() method in PyTorch.
	def __call__(self, x):
		fake_B = self.G_A2B(self.real_A)  # G_A(A)
		rec_A = self.G_B2A(self.fake_B)   # G_B(G_A(A))
		fake_A = self.G_B2A(self.real_B)  # G_B(B)
		rec_B = self.G_A2B(self.fake_A)   # G_A(G_B(B))
		return fake_B, rec_A, fake_A, rec_B



