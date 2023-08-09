"""
File contains the Decoder models.
"""
from abc import ABC

from flax import linen as nn
import jax.numpy as jnp


class Decoder(ABC, nn.Module):
    """Parent class for decoder model."""
    def __init__(self):
        super().__init__()


class MLPDecoder(Decoder):
    """
    MLP decoder model with the structure:

    z_tmp = Leaky_RELU(Dense(z))
    y = Dense(z_tmp)

    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = nn.Dense(self.hidden_dim, name="dec_hidden")(z)
        z = nn.gelu(z)
        z = nn.Dense(self.hidden_dim, name="dec_hidden2")(z)
        z = nn.gelu(z)
        z = nn.Dense(self.hidden_dim, name="dec_hidden3")(z)
        z = nn.gelu(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z


class ConvDecoder(Decoder):
    hidden_dim : int
    out_dim : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16*self.hidden_dim)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(features=2*self.hidden_dim, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2*self.hidden_dim, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=self.hidden_dim, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.hidden_dim, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2))(x)
        return x