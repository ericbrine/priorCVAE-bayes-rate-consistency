import hydra
from omegaconf import DictConfig, OmegaConf
import numpyro

import jax
from jax import random
import jax.numpy as jnp
import pandas as pd
import logging

import jax.config as config
numpyro.set_host_device_count(4)
config.update("jax_enable_x64", True)

from bayes_rate_consistency.simulation import load_simulated_dataset, sim_make_mcmc_data, simulation_inference, simulation_postprocess
from bayes_rate_consistency.simulation import get_output_path, save_simulated_data, save_mcmc_results


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = random.PRNGKey(cfg.seed)

    rng_key, rng_key_post, rng_key_predict, rng_key_draw = jax.random.split(rng_key, 4)

    output_dir = get_output_path(cfg.output_root, cfg.dataset.covid, cfg.dataset.size)

    if cfg.dataset.simulated:
        data = load_simulated_dataset(cfg.project_root, cfg.dataset.covid, cfg.dataset.size, cfg.dataset.strata)
        mcmc_data = sim_make_mcmc_data(data, cfg.dataset.strata)
        save_simulated_data(output_dir, cfg, data, mcmc_data)
        inference_data = simulation_inference(rng_key_post, rng_key_predict, cfg, mcmc_data, output_dir)
        simulation_postprocess(cfg, data, mcmc_data, inference_data, output_dir)



if __name__ == "__main__":
    main()
