import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

        
def simulation_model_vae(args, y: jnp.array = None, sample_lengthscale: bool = False):
    
    log_P = args['logP'] # A x A
    log_N = args['logN'] # A x A
    z_dim = args['z_dim']
    map_age_to_strata = args['map_age_to_strata']

    decoder = args['decoder']
    decoder_params = args['decoder_params']

    epsilon = 1e-13

    z = numpyro.sample("z", dist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))
    f = numpyro.deterministic("f", decoder.apply({'params': decoder_params}, z))
    f = jnp.reshape(f, (44,44))
    
    #fixed effects
    beta_0 = numpyro.sample('beta_0', dist.Normal(0, 10))
    v = numpyro.sample('v', dist.Exponential(1))
    # tau = numpyro.sample('tau', dist.Normal(0, 1))
    # rho = numpyro.sample('rho', dist.Normal(0, 1))

    log_contact_rate = numpyro.deterministic("log_contact_rate", beta_0 + f)
    log_m = log_contact_rate + log_P
    log_mu = log_m + log_N
    mu = jnp.exp(log_mu)
    alpha = mu/(v + epsilon)
    alpha_strata = numpyro.deterministic("alpha", jnp.matmul(alpha, map_age_to_strata))

    if y is None:
        numpyro.sample("yhat_strata", dist.GammaPoisson(alpha_strata, 1/v))
    else:
        numpyro.sample("y_strata", dist.GammaPoisson(alpha_strata, 1/v), obs=y)