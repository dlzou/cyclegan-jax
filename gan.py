""" Training and testing functions.

References:
    Learning rate scheduling docs:
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
    https://optax.readthedocs.io/en/latest/api.html

    Author Implementation: 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/cycle_gan_model.py
"""

from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
import optax

from image_pool import ImagePool
from networks import Discriminator, Generator, GanLoss, L1Loss
import config


class CycleGan():

    def __init__(self, options):
        # Initialize generators
        self.G_A = Generator() # Set parameters
        self.G_B = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

        self.criterion_gan = GanLoss()
        self.criterion_cycle = L1Loss()
        self.criterion_id = L1Loss()

        self.options = options
        
    def get_generator_params(self, rng, input_shape):
        """
        Reference: https://github.com/google/jax/issues/421
        """
        params_G_A = self.G_A.init(rng, jnp.ones(input_shape))["params"]
        params_G_B = self.G_B.init(rng, jnp.ones(input_shape))["params"]
        return (params_G_A, params_G_B)

    def get_discriminator_params(self, rng, input_shape):
        """
        Reference: https://github.com/google/jax/issues/421
        """
        params_D_A = self.D_A.init(rng, jnp.ones(input_shape))["params"]
        params_D_B = self.D_B.init(rng, jnp.ones(input_shape))["params"]
        return (params_D_A, params_D_B)

    def train_generator_forward(self, params, real_data):
        """
        Args:
            params: (params_G_A, params_G_B)
            real_data: (real_A, real_B)
        """
        params_G_A = params[0]
        params_G_B = params[1]

        real_A = real_data[0]
        real_B = real_data[1]
        
        # Forward through G
        fake_B = self.G_A.apply({"params": params_G_A}, real_A) # G_A(A)
        recover_A = self.G_B.apply({"params": params_G_B}, fake_B) # G_B(G_A(A))
        fake_A = self.G_B.apply({"params": params_G_B}, real_B) # G_B(B)
        recover_B = self.G_A.apply({"params": params_G_A}, fake_A) # G_A(G_B(B))
        
        return (fake_B, recover_A, fake_A, recover_B)
    
    def train_generator_backward(self, params, generated_data, real_data):
        lambda_A = self.options.lambda_A # loss weight on inputs from set A
        lambda_B = self.options.lambda_B # loss weight on inputs from set B
        lambda_id = self.options.lambda_id

        params_D_A = params[0]
        params_D_B = params[1]

        fake_B = generated_data[0]
        recover_A = generated_data[1]
        fake_A = generated_data[2]
        recover_B = generated_data[3]
        
        real_A = real_data[0]
        real_B = real_data[1]
        
        # Compute 3-criteria loss function
        
        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterion_gan(self.D_A.apply({"params": params_D_A}, fake_B))
        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterion_gan(self.D_B.apply({"params": params_D_B}, fake_A))

        # Cycle loss ||G_B(G_A(A)) - A||
        loss_cycle_A = self.criterion_cycle(recover_A, real_A) * lambda_A
        # Cycle loss ||G_A(G_B(B)) - B||
        loss_cycle_B = self.criterion_cycle(recover_B, real_B) * lambda_B

        # G_A2B should be identity if real_B is fed: ||G_A(B) - B||
        id_A = self.G_A(real_B)
        loss_id_A = self.criterion_id(id_A, real_B) * lambda_B * lambda_id
        # G_B2A should be identity if real_A is fed: ||G_B(A) - A||
        id_B = self.G_B(real_A)
        loss_id_B = self.criterion_id(id_B, real_A) * lambda_A * lambda_id
        
        return (
            loss_G_A + loss_G_B + 
            loss_cycle_A + loss_cycle_B + 
            loss_id_A + loss_id_B
        )

    # def forward_G(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     self.fake_B = self.G_A2B(self.real_A) # G_A(A)
    #     self.recover_A = self.G_B2A(self.fake_B) # G_B(G_A(A))
    #     self.fake_A = self.G_B2A(self.real_B) # G_B(B)
    #     self.recover_B = self.G_A2B(self.fake_A) # G_A(G_B(B))
    
    # def backward_G(self):
    #     lambda_A = self.hparams.lambda_A # loss weight on inputs from set A
    #     lambda_B = self.hparams.lambda_B # loss weight on inputs from set B
    #     lambda_id = self.hparams.lambda_id

    #     # GAN loss D_A(G_A(A))
    #     self.loss_G_A2B = self.criterion_gan(self.D_A(self.fake_B))
    #     # GAN loss D_B(G_B(B))
    #     self.loss_G_B2A = self.criterion_gan(self.D_B(self.fake_A))

    #     # Cycle loss ||G_B(G_A(A)) - A||
    #     self.loss_cycle_A = self.criterion_cycle(self.recover_A, self.real_A) * lambda_A
    #     # Cycle loss ||G_A(G_B(B)) - B||
    #     self.loss_cycle_B = self.criterion_cycle(self.recover_B, self.real_B) * lambda_B

    #     # G_A2B should be identity if real_B is fed: ||G_A(B) - B||
    #     self.id_A = self.G_A2B(self.real_B)
    #     self.loss_id_A = self.criterionIdt(self.id_A, self.real_B) * lambda_B * lambda_id
    #     # G_B2A should be identity if real_A is fed: ||G_B(A) - A||
    #     self.id_B = self.netG_B(self.real_A)
    #     self.loss_id_B = self.criterionIdt(self.id_B, self.real_A) * lambda_A * lambda_id
        
    #     self.loss_G = (
    #         self.loss_G_A2B + self.loss_G_B2A + 
    #         self.loss_cycle_A + self.loss_cycle_B + 
    #         self.loss_id_A + self.loss_id_B
    #     )

    # def loss_fn_D(self): 
    #     pred_real = self.D_A
    
    # def setup(self):
    #     self.G_A2B = None # Fill in 
    #     self.G_B2A = None # Fill in 

    #     if self.training: 
    #         self.D_A = None # Fill in 
    #         self.D_B = None # Fill in 

    #     if self.training:
    #         self.fake_A_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
    #         self.fake_B_pool = ImagePool(self.pool_size)  # create image buffer to store previously generated images
    
    # def train_step(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     # Train generator
        
    #     self.forward()      # compute fake images and reconstruction images.
        

    #     self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
    #     self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

    #     # self.backward_G()             # calculate gradients for G_A and G_B
    #     # self.optimizer_G.step()       # update G_A and G_B's weights
    #     self.state_G.apply_fn()
    #     self.state_G.apply_gradients()

    #     # Train discriminator
    #     self.set_requires_grad([self.netD_A, self.netD_B], True)
    #     self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
    #     self.backward_D_A()      # calculate gradients for D_A
    #     self.backward_D_B()      # calculate graidents for D_B
    #     self.optimizer_D.step()  # update D_A and D_B's weights

    # This is the equivalent of the forward() method in PyTorch.
    # def __call__(self, x):
    #     fake_B = self.G_A2B(self.real_A)  # G_A(A)
    #     rec_A = self.G_B2A(self.fake_B)   # G_B(G_A(A))
    #     fake_A = self.G_B2A(self.real_B)  # G_B(B)
    #     rec_B = self.G_A2B(self.fake_A)   # G_A(G_B(B))
    #     return fake_B, rec_A, fake_A, rec_B


def get_generator_state(model: CycleGan, learning_rate, beta_1):
    params_G = model.get_generator_params()
    tx = optax.adam(learning_rate, b1=beta_1)
    return TrainState.create(
        apply_fn=None, params=params_G, tx=tx,
    )


def get_discriminator_state(model: CycleGan, learning_rate, beta_1):
    params_G = model.get_discriminator_params()
    tx = optax.adam(learning_rate, b1=beta_1)
    return TrainState.create(
        apply_fn=None, params=params_G, tx=tx,
    )


@jax.jit
def generator_step(
    model: CycleGan,
    g_state: TrainState,
    d_state: TrainState,
    pool_data: ImagePool,
    real_data: jnp.ndarray,
):
    """The generator is updated by generating data and letting the discriminator
    critique it. It's loss goes down if the discriminator wrongly predicts it to
    to be real data.
    """
    def loss_fn(params):
        generated_data = model.train_generator_forward(params, real_data)
        # Update pool_data
        loss = model.train_generator_backward(d_state.params, pool_data, real_data)
        return loss, pool_data
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pool_data), grads = grad_fn(g_state.params)
    new_g_state = g_state.apply_gradients(grads=grads)
    return loss, new_g_state, pool_data


# @partial(jax.pmap, axis_name='num_devices')
def discriminator_step(g_state: TrainState,
                       d_state: TrainState,
                       real_data: jnp.ndarray,
                       rng: jnp.ndarray):
    """The discriminator is updated by critiquing both real and generated data,
    It's loss goes down as it predicts correctly if images are real or generated.
    """
    pass