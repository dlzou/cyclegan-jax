""" Training and testing functions.

References:
    Learning rate scheduling docs:
    https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
    https://optax.readthedocs.io/en/latest/api.html

    Author Implementation: 
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/cycle_gan_model.py
"""

from functools import partial
from typing import Sequence, Tuple

from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
import optax

from networks import Discriminator, Generator, GanLoss, L1Loss


class CycleGan:
    def __init__(self, opts):
        # The following are stateless; state is passed in later through params.
        self.G = Generator(
            opts.output_nc, opts.ngf, opts.n_res_blocks, opts.use_dropout
        )
        self.D = Discriminator(
            opts.ndf,
            opts.netD,
            opts.n_layers,
        )
        self.criterion_gan = GanLoss()
        self.criterion_cycle = L1Loss()
        self.criterion_id = L1Loss()

        self.lambda_A = opts.lambda_A  # weight of loss on inputs from set A
        self.lambda_B = opts.lambda_B  # weight of loss on inputs from set B
        self.lambda_id = opts.lambda_id  # weight of identity loss

    def get_generator_params(self, rng, input_shape):
        """
        Reference: https://github.com/google/jax/issues/421
        """
        params_G_A = self.G.init(rng, jnp.ones(input_shape))["params"]
        params_G_B = self.G.init(rng, jnp.ones(input_shape))["params"]
        return (params_G_A, params_G_B)

    def get_discriminator_params(self, rng, input_shape):
        params_D = self.D.init(rng, jnp.ones(input_shape))["params"]
        return params_D

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
        fake_B = self.G.apply({"params": params_G_A}, real_A)  # G_A(A)
        recover_A = self.G.apply({"params": params_G_B}, fake_B)  # G_B(G_A(A))
        fake_A = self.G.apply({"params": params_G_B}, real_B)  # G_B(B)
        recover_B = self.G.apply({"params": params_G_A}, fake_A)  # G_A(G_B(B))

        return (fake_B, recover_A, fake_A, recover_B)

    def train_generator_backward(self, params, generated_data, real_data):
        params_G_A = params[0]
        params_G_B = params[1]
        params_D_A = params[2]
        params_D_B = params[3]

        fake_B = generated_data[0]
        recover_A = generated_data[1]
        fake_A = generated_data[2]
        recover_B = generated_data[3]

        real_A = real_data[0]
        real_B = real_data[1]

        # Compute 3-criteria loss function

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterion_gan(
            self.D.apply({"params": params_D_A}, fake_B),
            target_is_real= True
        )  # ====================>
        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterion_gan(
            self.D.apply({"params": params_D_B}, fake_A),
            target_is_real=True
        )

        # Cycle loss ||G_B(G_A(A)) - A||
        loss_cycle_A = self.criterion_cycle(recover_A, real_A) * self.lambda_A
        # Cycle loss ||G_A(G_B(B)) - B||
        loss_cycle_B = self.criterion_cycle(recover_B, real_B) * self.lambda_B

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        id_A = self.G.apply({"params": params_G_A}, real_B)
        loss_id_A = self.criterion_id(id_A, real_B) * self.lambda_B * self.lambda_id
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        id_B = self.G.apply({"params": params_G_B}, real_A)
        loss_id_B = self.criterion_id(id_B, real_A) * self.lambda_A * self.lambda_id

        return loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B

    def train_discriminator_backward(self, params, real, fake):
        # Real
        pred_real = self.D.apply({"params", params}, real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        # TODO: CHANGE THIS DETACH
        # @source: https://github.com/google/jax/issues/2025
        # Idk what this means, should look more into this
        # pred_fake = netD(fake.detach())
        pred_fake = self.D.apply({"params": params}, jax.lax.stop_gradient(fake))
        loss_D_fake = self.criterion_gan(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


def create_generator_state(
    rng: jnp.ndarray,
    model: CycleGan,
    input_shape: Sequence[int],
    learning_rate: float,
    beta_1: float,
):
    params_G = model.get_generator_params(
        rng, input_shape
    )  # get params of both G_A and G_B
    tx = optax.adam(learning_rate, b1=beta_1)
    return TrainState.create(
        apply_fn=None,
        params=params_G,
        tx=tx,
    )


def create_discriminator_state(
    rng: jnp.ndarray,
    model: CycleGan,
    input_shape: Sequence[int],
    learning_rate: float,
    beta_1: float,
):
    params = model.get_discriminator_params(
        rng, input_shape
    )  # parameter for eithe G_A or G_B
    tx = optax.adam(learning_rate, b1=beta_1)
    return TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )


# @partial(jax.jit, static_argnums=0)
@jax.jit
def generator_step(
    model: CycleGan,
    g_state: TrainState,
    d_A_state: TrainState,
    d_B_state: TrainState,
    real_data: Tuple[jnp.ndarray, jnp.ndarray],
):
    """The generator is updated by generating data and letting the discriminator
    critique it. It's loss goes down if the discriminator wrongly predicts it to
    to be real data.
    """

    def loss_fn(params):  # param: g_state.params
        generated_data = model.train_generator_forward(
            params, real_data
        )  # ======================> fake, rec
        backward_params = (params[0], params[1], d_A_state.params, d_B_state.params)
        loss = model.train_generator_backward(
            backward_params, generated_data, real_data
        )  # =========================>
        return loss, generated_data

    grad_fn = jax.value_and_grad(
        loss_fn, has_aux=True
    )  # grad_fn has the same argument as loss_fn, but evaluate both loss_fn and grad of loss_fn
    (loss, generated_data), grads = grad_fn(g_state.params)
    new_g_state = g_state.apply_gradients(grads=grads)
    # what about metrics?
    return loss, new_g_state, generated_data


# @partial(jax.jit, static_argnums=0)
@jax.jit
def discriminator_step(
    model: CycleGan,
    d_A_state: TrainState,
    d_B_state: TrainState,
    real_data: Tuple[jnp.ndarray, jnp.ndarray],
    fake_data: Tuple[jnp.ndarray, jnp.ndarray],
):
    """The discriminator is updated by critiquing both real and generated data,
    It's loss goes down as it predicts correctly if images are real or generated.
    """

    # Step for D_A
    def loss_fn_A(params):
        loss = model.train_discriminator_backward(params, real_data, fake_data)
        return loss

    grad_fn = jax.value_and_grad(loss_fn_A)
    loss_A, grads = grad_fn(d_A_state.params)
    new_d_A_state = d_A_state.apply_gradients(grads=grads)

    # Step for D_B
    def loss_fn_B(params):
        loss = model.train_discriminator_backward(params, real_data, fake_data)
        return loss

    grad_fn = jax.value_and_grad(loss_fn_B)
    loss_B, grads = grad_fn(d_B_state.params)
    new_d_B_state = d_B_state.apply_gradients(grads=grads)

    return loss_A, loss_B, new_d_A_state, new_d_B_state
