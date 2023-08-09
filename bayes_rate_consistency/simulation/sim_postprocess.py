import arviz as az
import jax.numpy as jnp
import os

from bayes_rate_consistency.utils import plot_heatmap

def simulation_postprocess(cfg, data, mcmc_data, inference_data=None, output_dir=""):
    
    print("Postprocessing...")
    print("----------------------------------------")

    if not inference_data:
        inference_data = az.from_netcdf(os.path.join(output_dir, "mcmc.nc"))

    print("Extracting posterior contact intensities...")
    m = get_contact_intensity(data=data, part_gender='Male', contact_gender='Female')
    m_post = get_posterior_contact_intensity(inference_data, mcmc_data=mcmc_data, pop_key="P_F")

    mae = jnp.mean(jnp.abs(m - m_post))
    print(f"MAE: {mae}")

    print()
    print("Saving plots...")
    plot_heatmap(m.T, "True contact intensity", output_dir, filename="true_contact_intensity.png")
    plot_heatmap(m_post.T, "Posterior contact intensity", output_dir, filename="posterior_contact_intensity.png")




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
    ci_lower = jnp.quantile(yhat_strata, 0.025, axis=0)
    ci_upper = np.quantile(yhat_strata, 0.975, axis=0)
    in_range = (ci_lower <= y_strata) & (y_strata <= ci_upper)
    valid_idx = np.where(in_range)
    proportion = valid_idx[0].shape[0]/y_strata.size
    return proportion


def get_population(mcmc_data, key):
    P = mcmc_data[key]
    P = jnp.tile(P, (44, 1))
    return P