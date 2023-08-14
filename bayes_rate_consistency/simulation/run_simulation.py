from numpyro.infer import Predictive
import arviz as az
from hydra.utils import instantiate
import jax.numpy as jnp
from priorCVAE.priors import Matern52
import logging

from bayes_rate_consistency.mcmc import run_mcmc
from bayes_rate_consistency.decoder import load_decoder
from bayes_rate_consistency.model import simulation_model_vae
from bayes_rate_consistency.simulation import save_mcmc_results

log = logging.getLogger(__name__)

def simulation_inference(rng_key, rng_key_predict, cfg, mcmc_data, output_dir):

    log.info("Running MCMC...")
    log.info("----------------------------------------")

    args = {
        "logP": mcmc_data["log_P_M"],
        "logN": mcmc_data["log_N_M"],
        "map_age_to_strata": mcmc_data["map_age_to_strata"],
        "z_dim": 30,
        "num_warmup": cfg.model.num_warmup,
        "num_samples": cfg.model.num_samples,
        "num_chains": cfg.model.num_chains,
        "thinning": cfg.model.thinning,
    }

    if cfg.model.load_decoder:
        decoder, decoder_params = load_decoder(
            cfg.project_root,
            cfg.model.decoder_path,
            cfg.model.hidden_dim,
            cfg.model.input_dim,
        )
        args["decoder"] = decoder
        args["decoder_params"] = decoder_params
    else:
        A = mcmc_data["A"]
        x = jnp.linspace(0, 1, A)
        x1, x2 = jnp.meshgrid(x, x)
        x = jnp.stack([x1, x2], axis=-1)
        x1 = x1
        x2 = x2
        x = x.reshape([A ** 2, 2])
        args['kernel'] = Matern52(0.2)(x, x) + 1e-5 * jnp.eye(x.shape[0])
        args['size'] = x.shape[0]
        # args['x'] = x
    

    y = mcmc_data["Y_MM"]
    

    model_fn = instantiate(cfg.model.numpyro_model, _convert_="partial")
    
    # simulation_model_vae
    mcmc, mcmc_samples, t_elapsed = run_mcmc(
        rng_key=rng_key, model=model_fn, args=args, y=y
    )

    posterior_predictive = Predictive(model_fn, mcmc_samples)
    posterior_predictive_samples = posterior_predictive(rng_key_predict, args)
    
    log.info("Saving MCMC results...")
    log.info("----------------------------------------")
    inference_data = az.from_numpyro(mcmc, posterior_predictive=posterior_predictive_samples)
    save_mcmc_results(output_dir, inference_data)

    return inference_data
