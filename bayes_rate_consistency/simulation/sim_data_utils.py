import os
import pyreadr
import pandas as pd
import pickle
import arviz as az
import logging

from bayes_rate_consistency.utils import make_unique

log = logging.getLogger(__name__)


def load_simulated_dataset(project_root, intensity, size, strata):
    """
    Load simulated data from disk.
    
    :param project_root: project root directory.
    :param intensity: contact intensity.
    :param size: number of individuals.
    :param strata: age strata.
    :return: simulated data.
    """

    log.info("Loading simulated data...")
    log.info("----------------------------------------")
    datasets_dir = "data/simulations/datasets"

    data_dir = intensity

    data_dir += f"_{size}"
    data_dir += "_COVIMOD"
    data_path = os.path.join(project_root, datasets_dir, data_dir, "data_1.rds")

    log.info(f"Loading data from {data_path}")

    data = pyreadr.read_r(data_path)[None]
    data = categorize_age_strata(data)
    return data


def categorize_age_strata(data):
    """
    Categorize the 'alter_age_strata' column of the data frame.
    """
    age_order = ['6-9', '10-14', '15-19', '20-24', '25-34', '35-44', '45-49']
    data['alter_age_strata'] = pd.Categorical(data['alter_age_strata'], categories=age_order, ordered=True)
    return data


def save_simulated_data(output_dir, cfg, data, mcmc_data):
    """
    Save simulated data to disk.
    
    :param output_dir: output directory.
    :param cfg: configuration object.
    :param data: simulated data.
    :param mcmc_data: dictionary of data for MCMC.
    """

    data_df_path = make_unique(os.path.join(output_dir, "data.csv"))
    mcmc_data_path = make_unique(os.path.join(output_dir, "mcmc_data.pkl"))

    # save data
    data.to_csv(data_df_path)

    # save mcmc_data
    with open(mcmc_data_path, 'wb') as file:
        pickle.dump(mcmc_data, file)

    # save config
    cfg_path = make_unique(os.path.join(output_dir, "config.txt"))
    with open(cfg_path, 'w') as file:
        for key, value in cfg.items():
            file.write(f"{key}: {value}\n")
    

def get_output_path(project_root, is_covid, size, is_decoder = True):
    """
    Get the output path for simulated data.

    :param project_root: project root directory.
    :param is_covid: whether the data is COVID or pre-COVID.
    :param size: number of individuals.
    :param is_decoder: whether the model is a decoder or GP.
    :return: output path.
    """

    output_dir = "simulations"

    if is_covid:
        data_dir = "inCOVID"
    else:
        data_dir = "preCOVID"

    data_dir += f"_{size}"
    data_dir += "_COVIMOD"
    if is_decoder:
        data_dir += "_vae"
    else:
        data_dir += "_gp"

    data_dir = make_unique(os.path.join(project_root, output_dir, data_dir))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def save_mcmc_results(output_dir, az_data):
    """
    Save MCMC results to disk.

    :param output_dir: output directory.
    :param az_data: inference data.
    """
    mcmc_path = os.path.join(output_dir, "mcmc.nc")
    az_data.to_netcdf(mcmc_path)