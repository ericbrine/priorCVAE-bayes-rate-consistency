import arviz as az
import jax.numpy as jnp
import os
import logging

from bayes_rate_consistency.utils import plot_heatmap

log = logging.getLogger(__name__)

def simulation_postprocess(cfg, data, mcmc_data, inference_data=None, output_dir=""):
    
    log.info("Postprocessing...")
    log.info("----------------------------------------")

    if not inference_data:
        inference_data = az.from_netcdf(os.path.join(output_dir, "mcmc.nc"))

    log.info("Extracting posterior contact intensities...")
    m = get_contact_intensity(data=data, part_gender='Male', contact_gender='Male')
    m_post = get_posterior_contact_intensity(inference_data, mcmc_data=mcmc_data, pop_key="P_M")

    m_mae = jnp.mean(jnp.abs(m - m_post))
    log.info(f"MAE (contact intensity): {m_mae:.5f}")

    log.info("----------------------------------------")
    log.info("Posterior predictive check...")
    y_strata = mcmc_data['Y_MM']
    y_check = sim_posterior_predictive_check(y_strata, inference_data.posterior_predictive['yhat_strata'])
    log.info(f"Proportion of y in 95% CI: {y_check:.5f}")

    yhat_strata = inference_data.posterior_predictive.median(dim=["chain", "draw"])['yhat_strata'].values
    mae_y = jnp.mean(jnp.abs(y_strata - yhat_strata))
    log.info(f"MAE (y): {mae_y:.5f}")

    log.info("----------------------------------------")
    log.info("Saving plots...")
    plot_heatmap(m.T, "True contact intensity", output_dir, filename="true_contact_intensity.png", zmin=0, zmax=2)
    plot_heatmap(m_post.T, "Posterior contact intensity", output_dir, filename="posterior_contact_intensity.png", zmin=0, zmax=2)
    plot_heatmap(yhat_strata.T, "Y-hat", output_dir, filename="yhat_strata.png", color_scale='Magma')
    plot_heatmap(y_strata.T, "Y", output_dir, filename="y_strata.png", color_scale='Magma')


def get_posterior_contact_intensity(inference_data, mcmc_data, pop_key="P_M"):
    P = get_population(mcmc_data, pop_key)
    return jnp.exp(inference_data.posterior.median(dim=["chain", "draw"])['log_contact_rate'].values) * P


def get_contact_intensity(data, part_gender, contact_gender):
    m = data[
        (data['gender'] == part_gender) & 
        (data['alter_gender'] == contact_gender)
    ].sort_values(by=['age', 'alter_age']).cntct_intensity.to_numpy().reshape(44,44)
    return m


def sim_posterior_predictive_check(y_strata, yhat_strata):
    ci_lower = yhat_strata.quantile(0.025, dim=["chain", "draw"]).values
    ci_upper = yhat_strata.quantile(0.975, dim=["chain", "draw"]).values

    in_range = (ci_lower <= y_strata) & (y_strata <= ci_upper)
    num_valid = jnp.where(in_range)[0].size
    proportion = num_valid/y_strata.size
    return proportion


def get_population(mcmc_data, key):
    P = mcmc_data[key]
    P = jnp.tile(P, (44, 1))
    return P