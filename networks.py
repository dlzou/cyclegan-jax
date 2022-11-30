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
    
    def __init__(self, output_nc, ngf=64, n_res_blocks=6, use_dropout=True):
        """docstring"""
        super().__init__() # needed?
        model = [
            nn.Conv(features=ngf, kernel_size=[7, 7], padding=[(3, 3), (3, 3)]),
            nn.GroupNorm(group_size=1), # instance norm
            nn.relu,
        ]

        # Downsampling layers.
        n_downsample_layers = 2
        for i in range(n_downsample_layers):
            mult = 2 ** i
            model += [
                nn.Conv(features=ngf * mult * 2, kernel_size=[3, 3], strides=2, padding=[(1, 1), (1, 1)]),
                nn.GroupNorm(group_size=1), # instance norm
                nn.relu,
            ]
        
        # Resnet transformation blocks.
        mult = 2 ** n_downsample_layers
        for i in range(n_res_blocks):
            model += [ResnetBlock(ngf * mult, use_dropout)]
        
        # Upsampling layers.
        for i in range(n_downsample_layers):
            mult = 2 ** (n_downsample_layers - i)
            model += [
                nn.ConvTranspose(features=(ngf * mult) // 2, kernel_size=[3, 3], strides=2, padding=[(1, 1), (1, 1)]),
                nn.GroupNorm(group_size=1), # instance norm
                nn.relu,
            ]
        model += [nn.Conv(features=output_nc, kernel_size=[7, 7], padding=0)]
        model += [nn.activation.tanh]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, features, use_dropout):
        super().__init__()
        model = [
            nn.Conv(features=features, kernel_size=[3, 3], padding=[(1, 1), (1, 1)]),
            nn.GroupNorm(group_size=1), # instance norm
            nn.relu,
        ]
        if use_dropout:
            model += [nn.Dropout(0.5)]
        model += [
            nn.Conv(features=features, kernel_size=[3, 3], padding=[(1, 1), (1, 1)]),
            nn.GroupNorm(group_size=1), # instance norm
        ]
        self.model = nn.Sequential(*model)
    
    def __call__(self, input):
        return input + self.model(input)


class Discriminator(nn.Module):
    """
    The discriminator would take an image input and predict if it's an original 
    or the output from the generator.
    """
    def __init__(self, ndf, netD="n_layers", n_layers=3, norm='batch', init_type='normal', init_gain=0.02):
        """Initialize a Discriminator instance.

        Parameters:
            ndf (int)          -- the number of filters in the first conv layer
            netD (str)         -- the architecture's name: basic | n_layers | pixel
            n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
            norm (str)         -- the type of normalization layers used in the network.
            init_type (str)    -- the name of the initialization method.
            init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        """

        self.ndf = ndf
        self.netD = netD
        self.n_layers = n_layers
        self.norm = norm
        self.init_type = init_type
        self.init_gain = init_gain

    def setup(self):
        net = None
        #norm_layer = nn.GroupNorm(group_size=1) #only use groupnorm at this stage
        use_bias = False
        
        if self.netD == "n_layers" or "basic": #build N_layer discriminator
            kw, padw = 4, 1 #kernel width, padding width
            sequence = [nn.Conv(features=self.ndf, kernel_size=kw, strides=2, padding=padw), 
                        nn.PReLU(negative_slope_init=0.2)]
            nf_mult = 1
            #nf_mult_prev = 1
            for n in range(1, self.n_layers): #gradually increase the number of filters
                #nf_mult_prev = nf_mult
                nf_mult = jnp.min(2**n, 8)
                sequence += [nn.Conv(features=self.ndf*nf_mult, kernel_size=kw, strides=2, padding=padw,use_bias=use_bias),
                             nn.GroupNorm(group_size=1),
                             nn.PReLU(negative_slope_init=0.2)]
            
            #nf_mult_prev = nf_mult
            nf_mult = jnp.min(2**self.n_layers, 8)
            sequence += [nn.Conv(features=self.ndf*nf_mult, kernel_size=kw, strides=2, padding=padw, use_bias=use_bias),
                         nn.GroupNorm(group_size=1),
                         nn.PReLU(negative_slope_init=0.2)]
            
            sequence += [nn.Conv(1, kernel_size=kw, strides=1, padding=padw)]
            self.model = nn.Sequential(layers=sequence)
        
        elif self.netD == "pixel":
            sequence = [nn.Conv(features=self.ndf, kernel_size=1, stride=1, padding=0),
                        nn.PReLU(negative_slope_init=0.2),
                        nn.Conv(features=self.ndf*2, kernel_size=1, strides=1, padding=0, use_bias=use_bias),
                        nn.GroupNorm(group_size=1),
                        nn.PReLU(negative_slope_init=0.2),
                        nn.Conv(features=1, kernel_size=1, strides=1, padding=0, use_bias=use_bias)]
            self.model = nn.Sequential(layers=sequence)

        else:
            NotImplementedError('Discriminator model name [%s] is not recognized' % self.netD)
 
    def __call__(self, input):
        return self.model(input)




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