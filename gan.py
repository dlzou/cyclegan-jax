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
            output_nc=opts.output_nc,
            ngf=opts.ngf,
            n_res_blocks=opts.n_res_blocks,
            dropout_rate=opts.dropout_rate,
            initializer=opts.initializer,
        )
        self.D = Discriminator(
            ndf=opts.ndf,
            netD=opts.netD,
            n_layers=opts.n_layers,
            initializer=opts.initializer,
        )
        self.criterion_gan = GanLoss(gan_mode="lsgan")
        self.criterion_cycle = L1Loss()
        self.criterion_id = L1Loss()

        self.opts = opts
        self.lambda_A = opts.lambda_A  # weight of loss on inputs from set A
        self.lambda_B = opts.lambda_B  # weight of loss on inputs from set B
        self.lambda_id = opts.lambda_id  # weight of identity loss

    def get_generator_params(self, rngs, input_shape):
        """
        Reference: https://github.com/google/jax/issues/421
        """
        params_G_A = self.G.init(rngs, jnp.ones(input_shape), train=False)["params"]
        params_G_B = self.G.init(rngs, jnp.ones(input_shape), train=False)["params"]
        return (params_G_A, params_G_B)

    def get_discriminator_params(self, rngs, input_shape):
        params_D = self.D.init(rngs, jnp.ones(input_shape))["params"]
        return params_D

    def run_generator_forward(self, rngs, params, real_data, train=True):
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
        fake_B = self.G.apply(
            {"params": params_G_A}, real_A, train=train, rngs=rngs
        )  # G_A(A)
        recover_A = self.G.apply(
            {"params": params_G_B}, fake_B, train=train, rngs=rngs
        )  # G_B(G_A(A))
        fake_A = self.G.apply(
            {"params": params_G_B}, real_B, train=train, rngs=rngs
        )  # G_B(B)
        recover_B = self.G.apply(
            {"params": params_G_A}, fake_A, train=train, rngs=rngs
        )  # G_A(G_B(B))

        return (fake_B, recover_A, fake_A, recover_B)
    
    def run_single_generator_forward(self, rngs, params, real_data, direction="A"):
        """
        Args:
            params: (params_G_A, params_G_B)
            real_data: (real_A, real_B)
        """
        params_G_A = params[0]
        params_G_B = params[1]

        # Forward through G
        if direction == "A":
            fake = self.G.apply(
                {"params": params_G_A}, real_data, train=False, rngs=rngs
            )  # G_A(A)
            recover = self.G.apply(
                {"params": params_G_B}, fake, train=False, rngs=rngs
            )  # G_B(G_A(A))
        elif direction =="B":
            fake = self.G.apply(
                {"params": params_G_B}, real_data, train=False, rngs=rngs
            )  # G_A(A)
            recover = self.G.apply(
                {"params": params_G_A}, fake, train=False, rngs=rngs
            )  # G_B(G_A(A))
        else:
            raise ValueError("direction must be A or B")

        return (fake, recover)


    def run_generator_backward(
        self, rngs, params, generated_data, real_data, train=True
    ):
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
            self.D.apply({"params": params_D_A}, fake_B), target_is_real=True
        )
        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterion_gan(
            self.D.apply({"params": params_D_B}, fake_A), target_is_real=True
        )

        # Cycle loss ||G_B(G_A(A)) - A||
        loss_cycle_A = self.criterion_cycle(recover_A, real_A) * self.lambda_A
        # Cycle loss ||G_A(G_B(B)) - B||
        loss_cycle_B = self.criterion_cycle(recover_B, real_B) * self.lambda_B

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        id_A = self.G.apply({"params": params_G_A}, real_B, train=train, rngs=rngs)
        loss_id_A = self.criterion_id(id_A, real_B) * self.lambda_B * self.lambda_id
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        id_B = self.G.apply({"params": params_G_B}, real_A, train=train, rngs=rngs)
        loss_id_B = self.criterion_id(id_B, real_A) * self.lambda_A * self.lambda_id

        return loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B

    def run_discriminator_backward(self, params, real, fake):
        # Real
        pred_real = self.D.apply({"params": params}, real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        # TODO: CHANGE THIS DETACH
        # @source: https://github.com/google/jax/issues/2025
        # Idk what this means, should look more into this
        pred_fake = self.D.apply({"params": params}, jax.lax.stop_gradient(fake))
        loss_D_fake = self.criterion_gan(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


def create_generator_state(
    key: jnp.ndarray,
    model: CycleGan,
    input_shape: Sequence[int],
    learning_rate: float,
    beta_1: float,
):
    key, params_key = jax.random.split(key)
    key, dropout_key = jax.random.split(key)
    params_G = model.get_generator_params(
        {"params": params_key, "dropout": dropout_key}, input_shape
    )  # get params of both G_A and G_B
    tx = optax.adam(learning_rate, b1=beta_1)
    return key, TrainState.create(
        apply_fn=None,
        params=params_G,
        tx=tx,
    )


def create_discriminator_state(
    key: jnp.ndarray,
    model: CycleGan,
    input_shape: Sequence[int],
    learning_rate: float,
    beta_1: float,
):
    key, params_key = jax.random.split(key)
    params = model.get_discriminator_params(
        {"params": params_key}, input_shape
    )  # parameter for eithe G_A or G_B
    tx = optax.adam(learning_rate, b1=beta_1)
    return key, TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )


@partial(jax.jit, static_argnums=1)
def generator_step(
    key: jnp.ndarray,
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
    key, dropout_key = jax.random.split(key)

    def loss_fn(params):  # param: g_state.params
        generated_data = model.run_generator_forward(
            {"dropout": dropout_key}, params, real_data, train=True
        )
        backward_params = (params[0], params[1], d_A_state.params, d_B_state.params)
        loss = model.run_generator_backward(
            {"dropout": dropout_key},
            backward_params,
            generated_data,
            real_data,
            train=True,
        )
        return loss, generated_data

    grad_fn = jax.value_and_grad(
        loss_fn, has_aux=True
    )  # grad_fn has the same argument as loss_fn, but evaluate both loss_fn and grad of loss_fn
    (loss, generated_data), grads = grad_fn(g_state.params)
    new_g_state = g_state.apply_gradients(grads=grads)
    # what about metrics?
    return key, loss, new_g_state, generated_data


@partial(jax.jit, static_argnums=1)
def generator_validation(
    key: jnp.ndarray,
    model: CycleGan,
    g_state: TrainState,
    d_A_state: TrainState,
    d_B_state: TrainState,
    real_data: Tuple[jnp.ndarray, jnp.ndarray],
):
    key, dropout_key = jax.random.split(key)

    generated_data = model.run_generator_forward(
        {"dropout": dropout_key}, g_state.params, real_data, train=False
    )
    backward_params = (
        g_state.params[0],
        g_state.params[1],
        d_A_state.params,
        d_B_state.params,
    )
    loss = model.run_generator_backward(
        {"dropout": dropout_key},
        backward_params,
        generated_data,
        real_data,
        train=False,
    )
    return key, loss, generated_data


@partial(jax.jit, static_argnums=0)
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
        loss = model.run_discriminator_backward(params, real_data[1], fake_data[1])
        return loss

    grad_fn = jax.value_and_grad(loss_fn_A)
    loss_A, grads = grad_fn(d_A_state.params)
    new_d_A_state = d_A_state.apply_gradients(grads=grads)

    # Step for D_B
    def loss_fn_B(params):
        loss = model.run_discriminator_backward(params, real_data[0], fake_data[0])
        return loss

    grad_fn = jax.value_and_grad(loss_fn_B)
    loss_B, grads = grad_fn(d_B_state.params)
    new_d_B_state = d_B_state.apply_gradients(grads=grads)

    return loss_A, loss_B, new_d_A_state, new_d_B_state


@partial(jax.jit, static_argnums=[1, 4])
def generator_prediction(
    key: jnp.ndarray,
    model: CycleGan,
    g_state: TrainState,
    real_data: jnp.ndarray,
    direction: str,
):
    key, dropout_key = jax.random.split(key)

    generated_data = model.run_single_generator_forward(
        {"dropout": dropout_key}, g_state.params, real_data, direction=direction
    )
    return key, generated_data
