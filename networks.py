""" Network components used by GAN.

References:
    Author's implementation in PyTorch:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py

    Blog post:
    https://hardikbansal.github.io/CycleGANBlog/
"""

from functools import partial
from typing import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp


class Generator(nn.Module):
    """The generator is responsible for generating images that attempt to fool
    the discriminator.

    Args:
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        n_res_blocks (int)  -- the number of ResNet blocks
        drop_rate (float)   -- the rate for dropout sampling
        upsample_mode (str) -- deconv or bilinear upsampling
        initializer (fn)    -- function for initializing parameters

    Reference for upsampling:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675
    """

    output_nc: int = 3
    ngf: int = 32
    n_res_blocks: int = 6
    dropout_rate: float = 0.5
    upsample_mode: str = "deconv"
    initializer: Callable = jax.nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, x, train):
        input_x = x
        # First convolution layer.
        first_conv = nn.Sequential(
            [
                nn.Conv(
                    features=self.ngf,
                    kernel_size=[7, 7],
                    strides=[1, 1],
                    padding="SAME",
                    kernel_init=self.initializer,
                ),
                nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
                nn.relu,
            ]
        )
        x = first_conv(x)

        # Downsampling layers.
        n_downsample_layers = 2
        model = []
        for i in range(n_downsample_layers):
            mult = 2**i
            model += [
                nn.Conv(
                    features=self.ngf * mult * 2,
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    padding="SAME",
                    kernel_init=self.initializer,
                ),
                nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
                nn.relu,
            ]
        downsample = nn.Sequential(model)
        x = downsample(x)

        # Resnet transformation blocks.
        mult = 2**n_downsample_layers
        model = []
        for i in range(self.n_res_blocks):
            model += [
                ResnetBlock(
                    features=self.ngf * mult,
                    dropout_layer=partial(
                        nn.Dropout, rate=self.dropout_rate, deterministic=not train
                    ),
                    initializer=self.initializer,
                )
            ]
        transform = nn.Sequential(model)
        x = transform(x)

        # Upsampling layers.
        model = []
        if self.upsample_mode == "bilinear":
            for i in range(n_downsample_layers):
                mult = 2 ** (n_downsample_layers - i)
                model += [
                    partial(
                        jax.image.resize,
                        shape=(
                            x.shape[0],
                            x.shape[1] * 2 ** (i + 1),
                            x.shape[2] * 2 ** (i + 1),
                            x.shape[3],
                        ),
                        method="bilinear",
                    ),
                    nn.Conv(
                        features=(self.ngf * mult) // 2,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="SAME",
                        kernel_init=self.initializer,
                    ),
                    nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
                    nn.relu,
                ]
        elif self.upsample_mode == "deconv":
            for i in range(n_downsample_layers):
                mult = 2 ** (n_downsample_layers - i)
                model += [
                    nn.ConvTranspose(
                        features=(self.ngf * mult) // 2,
                        kernel_size=[3, 3],
                        strides=[2, 2],
                        padding="SAME",
                        kernel_init=self.initializer,
                    ),
                    nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
                    nn.relu,
                ]
        else:
            NotImplementedError(
                "Generator upsample_mode [%s] is not recognized" % self.upsample_mode
            )
        upsample = nn.Sequential(model)
        x = upsample(x)

        # Last convolution layer.
        model = [
            nn.Conv(
                features=self.output_nc,
                kernel_size=[7, 7],
                strides=[1, 1],
                padding="SAME",
                kernel_init=self.initializer,
            )
        ]
        model += [nn.activation.tanh]
        last_conv = nn.Sequential(model)
        x = last_conv(x)

        # Add skip connection between generator input and output.
        # Reference: https://github.com/leehomyc/cyclegan-1
        # return x + input_x
        return x


class ResnetBlock(nn.Module):

    features: int
    dropout_layer: Callable
    initializer: Callable = jax.nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, x):
        model = [
            nn.Conv(
                features=self.features,
                kernel_size=[3, 3],
                padding="SAME",
                kernel_init=self.initializer,
            ),
            nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
            nn.relu,
            self.dropout_layer(),
            nn.Conv(
                features=self.features,
                kernel_size=[3, 3],
                padding="SAME",
                kernel_init=self.initializer,
            ),
            nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
        ]
        seq_model = nn.Sequential(model)

        return x + seq_model(x)


class Discriminator(nn.Module):
    """
    The discriminator would take an image input and predict if it's an original
    or the output from the generator.

    Parameters:
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        init_type (str)    -- the name of the initialization method.
    """

    ndf: int = 64  # TODO: What's this??
    netD: str = "n_layers"
    n_layers: int = 3
    initializer: Callable = jax.nn.initializers.normal(stddev=0.02)

    def setup(self):
        net = None
        use_bias = False

        if self.netD == "n_layers" or "basic":  # build N_layer discriminator
            kw, padw = 4, 1  # kernel width, padding width
            sequence = [
                nn.Conv(
                    features=self.ndf,
                    kernel_size=[kw, kw],
                    strides=2,
                    padding=padw,
                    kernel_init=self.initializer,
                ),
                nn.PReLU(negative_slope_init=0.2),
            ]
            nf_mult = 1
            # nf_mult_prev = 1
            for n in range(
                1, self.n_layers
            ):  # gradually increase the number of filters
                # nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    nn.Conv(
                        features=self.ndf * nf_mult,
                        kernel_size=[kw, kw],
                        strides=2,
                        padding=padw,
                        use_bias=use_bias,
                        kernel_init=self.initializer,
                    ),
                    nn.GroupNorm(num_groups=None, group_size=1),
                    nn.PReLU(negative_slope_init=0.2),
                ]

            # nf_mult_prev = nf_mult
            nf_mult = min(2**self.n_layers, 8)
            sequence += [
                nn.Conv(
                    features=self.ndf * nf_mult,
                    kernel_size=[kw, kw],
                    strides=2,
                    padding=padw,
                    use_bias=use_bias,
                    kernel_init=self.initializer,
                ),
                nn.GroupNorm(num_groups=None, group_size=1),
                nn.PReLU(negative_slope_init=0.2),
            ]

            sequence += [nn.Conv(1, kernel_size=[kw, kw], strides=1, padding=padw)]
            self.model = nn.Sequential(layers=sequence)

        elif self.netD == "pixel":
            sequence = [
                nn.Conv(
                    features=self.ndf,
                    kernel_size=[1, 1],
                    stride=1,
                    padding=0,
                    kernel_init=self.initializer,
                ),
                nn.PReLU(negative_slope_init=0.2),
                nn.Conv(
                    features=self.ndf * 2,
                    kernel_size=[1, 1],
                    strides=1,
                    padding=0,
                    use_bias=use_bias,
                    kernel_init=self.initializer,
                ),
                nn.GroupNorm(num_groups=None, group_size=1),
                nn.PReLU(negative_slope_init=0.2),
                nn.Conv(
                    features=1,
                    kernel_size=[1, 1],
                    strides=1,
                    padding=0,
                    use_bias=use_bias,
                    kernel_init=self.initializer,
                ),
            ]
            self.model = nn.Sequential(layers=sequence)

        else:
            NotImplementedError(
                "Discriminator model name [%s] is not recognized" % self.netD
            )

    def __call__(self, input):
        return self.model(input)


class GanLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        # super(self).__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label  # related to register_buffer()
        self.target_fake_label = target_fake_label

    def setup(self):
        if self.gan_mode not in ["lsgan", "vanilla", "wgangp"]:
            raise NotImplementedError("gan mode %s not implemented" % self.gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label arrays with the same size as the input.

        Parameters:
            prediction (jnp.ndarray) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label array filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_label = self.target_real_label
        else:
            target_label = self.target_fake_label
        target = jnp.ones_like(a=prediction)
        return target_label * target

    def __call__(self, prediction, target_is_real):
        target_array = self.get_target_tensor(prediction, target_is_real)
        if self.gan_mode in ["lsgan"]:  # use MSELoss
            if target_array.shape != prediction.shape:
                raise ValueError(
                    "Target size ({}) must be the same as input size ({})".format(
                        target_array.shape, prediction.shape
                    )
                )
            loss_value = jnp.mean((prediction - target_array) ** 2)

        elif self.gan_mode in ["vanilla"]:  # use BCEWithLogitsLoss
            # source: https://github.com/deepchem/jaxchem/blob/master/jaxchem/loss/binary_cross_entropy_with_logits.py#L4-L37
            if target_array.shape != prediction.shape:
                raise ValueError(
                    "Target size ({}) must be the same as input size ({})".format(
                        target_array.shape, prediction.shape
                    )
                )
            max_val = jnp.clip(-prediction, 0, None)
            loss_value = (
                prediction
                - prediction * target_array
                + max_val
                + jnp.log(jnp.exp(-max_val) + jnp.exp((-prediction - max_val)))
            )
            loss_value = jnp.mean(loss_value)  # default to mean loss

        elif self.gan_mode in ["wgangp"]:
            if target_is_real:
                loss_value = -jnp.mean(prediction)
            else:
                loss_value = jnp.mean(prediction)
        return loss_value

    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py#L210


class L1Loss(nn.Module):
    """
    Simple L1 Loss, optax doesn't have L1 loss.
    """

    def __init__(self):
        # super().__init__()
        pass

    def __call__(self, prediction, target_array):
        if target_array.shape != prediction.shape:
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target_array.shape, prediction.shape
                )
            )
        absolute = jnp.abs(target_array - prediction)
        return jnp.mean(absolute)
