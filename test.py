import random as rnd

import numpy as np 
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
numpyro.set_host_device_count(4)
import jax
import optax
from jax import random
import jax.numpy as jnp
from flax import linen as nn

from priorCVAE.models import MLPEncoder, MLPDecoder, VAE
from priorCVAE.priors import SquaredExponential, Matern32, Matern52
from priorCVAE.datasets import GPDataset
from priorCVAE.trainer import VAETrainer
from priorCVAE.losses import SquaredSumAndKL
from priorCVAE.utility import save_model_params, load_model_params

import jax.config as config
config.update("jax_enable_x64", True)

numpyro.enable_validation()
kernel = Matern52(lengthscale=0.5)

gp_set = GPDataset(n_data=1936, kernel=kernel, sample_lengthscale=False, data_dim=2) 
rng_key, _ = random.split(random.PRNGKey(1)) # random.split(random.PRNGKey(rnd.randint(0, 9999)))
gp_set.reshape = False
print(gp_set.x.shape)
breakpoint()
x_test, y_test, ls_test = gp_set.simulatedata(n_samples=1)