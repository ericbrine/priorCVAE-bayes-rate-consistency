import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

from priorCVAE.priors import Matern52
# from mlkernels.jax import Matern52 as Matern52_ml
# from lab import B


def simulation_model_gp(args, y = None):
    
    log_P = args['logP'] # A x A
    log_N = args['logN'] # A x A
    map_age_to_strata = args['map_age_to_strata']

    epsilon = 1e-13
    jitter = 1e-6

    k = args['kernel']
    size = args['size']

    f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(size), covariance_matrix=k))
    f = jnp.reshape(f, (44,44))
    # Symmetrize
    i_lower = jnp.tril_indices(44, -1)
    f = f.at[i_lower].set(f.T[i_lower])
   
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


# def simulation_model_gp(args, y = None):
    
#     log_P = args['logP'] # A x A
#     log_N = args['logN'] # A x A
#     map_age_to_strata = args['map_age_to_strata']

#     epsilon = 1e-13
#     jitter = 1e-6

#     kernel = args['kernel']
#     size = args['size']

#     # kernel.variance = 2.0

#     x1 = jnp.linspace(0, 1, 44)
#     # kernel.lengthscale = 0.4
#     kx = Matern52(0.4, 2.0)(x1, x1)
#     # kernel.lengthscale = 2.0
#     ky = Matern52(2.0, 2.0)(x1, x1)
#     k = jnp.kron(ky, kx).T

   
#     # kx = Matern52_ml().stretch(0.4)(x1, x1)
#     # ky = Matern52_ml().stretch(2.0)(x1, x1)
#     # kx = B.dense(kx)
#     # ky = B.dense(ky)
#     # k = jnp.kron(kx, ky)
#     k += jitter * jnp.eye(size)

#     f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(size), covariance_matrix=k))
#     f = jnp.reshape(f, (44,44))
#     # Symmetrize
#     i_lower = jnp.tril_indices(44, -1)
#     f = f.at[i_lower].set(f.T[i_lower])
   
#     #fixed effects
#     beta_0 = numpyro.sample('beta_0', dist.Normal(0, 10))
#     v = numpyro.sample('v', dist.Exponential(1))
#     # tau = numpyro.sample('tau', dist.Normal(0, 1))
#     # rho = numpyro.sample('rho', dist.Normal(0, 1))

#     log_contact_rate = numpyro.deterministic("log_contact_rate", beta_0 + f)
#     log_m = log_contact_rate + log_P
#     log_mu = log_m + log_N
#     mu = jnp.exp(log_mu)
#     alpha = mu/(v + epsilon)
#     alpha_strata = numpyro.deterministic("alpha", jnp.matmul(alpha, map_age_to_strata))

#     if y is None:
#         numpyro.sample("yhat_strata", dist.GammaPoisson(alpha_strata, 1/v))
#     else:
#         numpyro.sample("y_strata", dist.GammaPoisson(alpha_strata, 1/v), obs=y)


# def simulation_model_gp(args, y = None):
    
#     log_P = args['logP'] # A x A
#     log_N = args['logN'] # A x A
#     map_age_to_strata = args['map_age_to_strata']

#     epsilon = 1e-13
#     jitter = 1e-6

#     length = numpyro.sample("gp_lengthscale", dist.HalfCauchy(1.))
    
#     # compute kernel
#     x = args['x']
#     k = args['kernel'].stretch(length)(x, x)
#     k = B.dense(k)

#     k += jitter * jnp.eye(x.shape[0])

#     f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
#     f = jnp.reshape(f, (44,44))
   
#     #fixed effects
#     beta_0 = numpyro.sample('beta_0', dist.Normal(0, 10))
#     v = numpyro.sample('v', dist.Exponential(1))
#     # tau = numpyro.sample('tau', dist.Normal(0, 1))
#     # rho = numpyro.sample('rho', dist.Normal(0, 1))

#     log_contact_rate = numpyro.deterministic("log_contact_rate", beta_0 + f)
#     log_m = log_contact_rate + log_P
#     log_mu = log_m + log_N
#     mu = jnp.exp(log_mu)
#     alpha = mu/(v + epsilon)
#     alpha_strata = numpyro.deterministic("alpha", jnp.matmul(alpha, map_age_to_strata))

#     if y is None:
#         numpyro.sample("yhat_strata", dist.GammaPoisson(alpha_strata, 1/v))
#     else:
#         numpyro.sample("y_strata", dist.GammaPoisson(alpha_strata, 1/v), obs=y)