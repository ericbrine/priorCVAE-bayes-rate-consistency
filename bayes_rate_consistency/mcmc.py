"""
File contains the code for Monte Carlo Markov Chain (MCMC) used for inference.
"""
from typing import Dict
import time
import os
import logging

import numpy as np
from jax.random import KeyArray
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
from numpyro.infer import init_to_median, MCMC, NUTS

from priorCVAE.models import Decoder


log = logging.getLogger(__name__)

def run_mcmc(rng_key: KeyArray, model: numpyro.primitives, args: Dict, y: jnp.array = None, verbose: bool = False) -> [MCMC, jnp.ndarray, float]:
    """
    Run MCMC inference using VAE decoder.

    :param rng_key: a PRNG key used as the random key.
    :param model: a numpyro model of the type numpypro primitives.
    :param args: a dictionary with the arguments required for MCMC.
    :param decoder: a decoder model.
    :param decoder_params: a dictionary with decoder network parameters.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).
    :param verbose: if True, logs the MCMC summary.

    Returns:
        - MCMC object
        - MCMC samples
        - time taken

    """
    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        chain_method="parallel",
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    start = time.time()
    mcmc.run(rng_key, args, y)
    t_elapsed = time.time() - start
    if verbose:
        mcmc.log.info_summary(exclude_deterministic=False)

    log.info(f"MCMC elapsed time: {t_elapsed:.2f}s")
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    r = np.mean(ss['f']['n_eff'])
    log.info("Average ESS for all VAE-GP effects : " + str(round(r)))

    return mcmc, mcmc.get_samples(), t_elapsed
