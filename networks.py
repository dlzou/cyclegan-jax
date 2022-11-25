""" Network components used by GAN.

References:
    Author's implementation in PyTorch:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py

    Blog post:
    https://hardikbansal.github.io/CycleGANBlog/
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class Generator(nn.Module):
    """
    The generator would...
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6):
        """docstring"""
        super().__init__() # needed?
        model = [
            nn.Conv(features=ngf, kernel_size=[7, 7], padding=[(3, 3), (3, 3)]),
            nn.GroupNorm(group_size=1), # instance norm
            nn.relu,
        ]
        self.model = nn.Sequential(*model)

        # Downsampling layers.
        n_downsample_layers = 2
        for i in range(n_downsample_layers):
            mult = 2 ** i
            model += [
                nn.Conv(features=ngf * mult * 2, kernel_size=[3, 3], padding=[(1, 1), (1, 1)]),
                nn.GroupNorm(group_size=1), # instance norm nn.relu,
                nn.relu,
            ]
        
        # Resnet transformation blocks.
        
        # Upsampling layers.
    
    def forward(self, input):
        return self.mdoel(input)


class ResnetBlock(nn.Module):

    def __init__(self):
        pass


class Discriminator(nn.Module):
    """
    The discriminator would take an image input and predict if it's an original 
    or the output from the generator.
    """
    def __init__(self, input_nc, ndf, netD="n_layers", n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
        """Initialize a Discriminator instance.

        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            netD (str)         -- the architecture's name: basic | n_layers | pixel
            n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
            norm (str)         -- the type of normalization layers used in the network.
            init_type (str)    -- the name of the initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        """
        net = None
        norm_layer = nn.GroupNorm(group_size=1) #only use groupnorm at this stage
        use_bias = False
        
        if netD == "n_layers" or "basic": #build N_layer discriminator
            kw, padw = 4, 1 #kernel width, padding width
            sequence = [nn.Conv(input_nc, kw, 2, padw), nn.PReLU(negative_slope_init=0.2)]
    pass




class GanLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    
    # TODO: Add gan_mode later
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
                """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        
    # TODO: Complete GAN Loss, reference here: 
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py#L210

class L1Loss(nn.Module):
    """
    Simple L1 Loss, optax doesn't have L1 loss. 
    """
    def __init__():
        pass 