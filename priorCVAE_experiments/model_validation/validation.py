"""
File contains tests for validating VAE models.
"""

import jax.numpy as jnp
import numpy as np

from priorCVAE.diagnostics import frobenius_norm_of_diff, sample_covariance
from priorCVAE.priors import Kernel, SquaredExponential
from .utils import mean_bootstrap_interval


def bootstrap_mean_test(samples: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Test if zero lies within the 5th and 95th quantiles of the bootstrap distribution of the mean.

    :param samples: samples to be tested.
    :return: Proportion of bootstrap intervals containing zero.
    """
    ci_lower, ci_upper = mean_bootstrap_interval(samples)
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = jnp.where(zero_in_interval)[0].shape[0]
    return num_valid / samples.shape[1]


def norm_of_kernel_diff(samples: jnp.ndarray, kernel: Kernel, grid: jnp.array, **kwargs) -> jnp.ndarray:
    """
    Calculate the norm of the difference of the empirical covariance matrix and the kernel covariance.

    :param samples: samples used to compute empirical covariance.
    :param kernel: kernel used to compute covariance.
    :param grid: grid used to compute kernel covariance.
    :return: norm of the difference between the sample covariance and the kernel.
    """
    cov = sample_covariance(samples)
    norm = frobenius_norm_of_diff(cov, kernel(grid, grid))
    return norm


def bootstrap_covariance_test(samples: jnp.ndarray, kernel: Kernel, grid: jnp.array, sample_size=4000, num_iterations=1000, **kwargs):
    """
    Test if the kernel matrix lies within the 5th and 95th quantiles of the bootstrap distribution of the sample covariance matrix.

    :param samples: samples to be tested.
    :param kernel: kernel used to compute covariance.
    :param grid: grid used to compute kernel covariance.
    :param sample_size: size of each bootstrap sample.
    :param num_iterations: number of bootstrap samples.
    :return: proportion of points at which the test passes.
    """
    
    stats = []
    n = samples.shape[0]
    for _ in range(num_iterations):
        bootstrap_sample = samples[np.random.choice(n, size=sample_size, replace=True)]
        stat = sample_covariance(bootstrap_sample).flatten()
        stats.append(stat)

    stats = np.array(stats)

    ci_lower = np.percentile(stats, 5, axis=0)
    ci_upper = np.percentile(stats, 95, axis=0)

    K = kernel(grid,grid).flatten()

    in_range = (ci_lower <= K) & (K <= ci_upper)
    valid_idx = np.where(in_range)
    
    num_valid = valid_idx[0].shape[0]
    total = in_range.shape[0]

    return num_valid/total

def bootstrap_isotropy_test(samples: jnp.ndarray, sample_size=4000, num_iterations=1000, **kwargs):
    """
    Test the isotropic properties of the VAE samples.

    :param samples: samples to be tested.
    :param sample_size: size of each bootstrap sample.
    :param num_iterations: number of bootstrap samples.
    :return: proportion of points at which the test passes.
    """
    
    stats = []
    n = samples.shape[0]
    for _ in range(num_iterations):
        bootstrap_sample = samples[np.random.choice(n, size=sample_size, replace=True)]
        _, diffs = compute_covariances_at_distances(bootstrap_sample)

        stat = diffs
        stats.append(stat)

    stats = np.array(stats)

    ci_lower = np.percentile(stats, 2.5, axis=0)
    ci_upper = np.percentile(stats, 97.5, axis=0)

    in_range = (ci_lower <= 0) & (0 <= ci_upper)
    valid_idx = np.where(in_range)
    
    num_valid = valid_idx[0].shape[0]
    total = in_range.shape[0]
    return num_valid/total



def compute_covariances_at_distances(flattened_samples, grid_size=25):

    # Compute empirical covariance matrix
    empirical_cov_matrix = np.cov(flattened_samples, rowvar=False)

    x = np.linspace(0, 1, grid_size)
    ii, jj = np.meshgrid(x, x, indexing='ij')
    coordinates = np.column_stack([ii.ravel(), jj.ravel()])
    pairwise_diff = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)

    tolerance = 1e-2
    rounded_diff = np.round(pairwise_diff / tolerance) * tolerance

    # Create dictionary to hold covariances for each unique distance
    grouped_covs = {}
    for unique_distance in np.unique(rounded_diff):
        indices = np.where(rounded_diff == unique_distance)
        covs_for_this_distance = empirical_cov_matrix[indices]
        if len(covs_for_this_distance) > 5:  # Ensure at least 6 points for 5 covariances
            grouped_covs[unique_distance] = covs_for_this_distance

    # Compute first-order differences for covariances of each unique distance
    diffs = {}
    for distance, covs in grouped_covs.items():
        diffs[distance] = np.diff(covs)

    # Flatten the diffs dictionary into a single array
    flattened_diffs = np.array([np.mean(diff_array) for diff_array in diffs.values()])

    return grouped_covs, flattened_diffs