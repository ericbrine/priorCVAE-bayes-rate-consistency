import pandas as pd
import numpy as np
import jax.numpy as jnp
import re
import logging

from bayes_rate_consistency.simulation import save_simulated_data

log = logging.getLogger(__name__)


def sim_make_mcmc_data(data, strata_scheme, A=44):
    """
    Make the data dictionary for MCMC.

    :param data: simulated data.
    :param strata_scheme: age strata scheme.
    :param A: number of age strata.
    :return: dictionary of data for MCMC.
    """

    data = categorize_age_strata(data)
    mcmc_data = init_mcmc_data(A, strata_scheme)
    mcmc_data = add_N(mcmc_data, data)
    mcmc_data = add_contacts(mcmc_data, data)

    mcmc_data = add_partsize_offsets(mcmc_data, data)
    mcmc_data = add_pop_offsets(mcmc_data, data)
    mcmc_data = add_age_strata_map(mcmc_data, strata_scheme)

    return mcmc_data


def categorize_age_strata(data):
    """
    Categorize the 'alter_age_strata' column of the data frame.
    """

    age_order = ['6-9', '10-14', '15-19', '20-24', '25-34', '35-44', '45-49']
    # Convert the 'alter_age_strata' column to a categorical type with the custom order
    data['alter_age_strata'] = pd.Categorical(data['alter_age_strata'], categories=age_order, ordered=True)
    return data

def init_mcmc_data(A, strata_scheme):
    """
    Initialize the data dictionary for MCMC.
    """

    strata_config = {
        "COVIMOD": 7,
        "CoMix": 5,
        "5yr": 9,
        "3yr": 15
    }
    return {'A': A, 'C': strata_config[strata_scheme]}


def add_N(mcmc_data, data):
    """
    Add the number of contacts to the data dictionary for MCMC.
    """
    unique_data = data.copy().drop_duplicates(subset=['age', 'gender', 'alter_age_strata', 'alter_gender'])
    mcmc_data['N_MM'] = len(unique_data[(unique_data['gender'] == 'Male') & (unique_data['alter_gender'] == 'Male')])
    mcmc_data['N_FF'] = len(unique_data[(unique_data['gender'] == 'Female') & (unique_data['alter_gender'] == 'Female')])
    mcmc_data['N_MF'] = len(unique_data[(unique_data['gender'] == 'Male') & (unique_data['alter_gender'] == 'Female')])
    mcmc_data['N_FM'] = len(unique_data[(unique_data['gender'] == 'Female') & (unique_data['alter_gender'] == 'Male')])
    return mcmc_data


def add_contacts(mcmc_data, data):
    """
    Add the contact matrix to the data dictionary for MCMC.
    """
    unique_data = data.drop_duplicates(subset=['age', 'gender', 'alter_age_strata', 'alter_gender', 'y_strata'])
    unique_data = unique_data.sort_values(['age', 'alter_age_strata'])
    mcmc_data['Y_MM'] = unique_data[(unique_data['gender'] == 'Male') & (unique_data['alter_gender'] == 'Male')]['y_strata']
    mcmc_data['Y_FF'] = unique_data[(unique_data['gender'] == 'Female') & (unique_data['alter_gender'] == 'Female')]['y_strata']
    mcmc_data['Y_MF'] = unique_data[(unique_data['gender'] == 'Male') & (unique_data['alter_gender'] == 'Female')]['y_strata']
    mcmc_data['Y_FM'] = unique_data[(unique_data['gender'] == 'Female') & (unique_data['alter_gender'] == 'Male')]['y_strata']
    
    mcmc_data['Y_MM'] = jnp.array(mcmc_data['Y_MM'].to_numpy().reshape(mcmc_data['A'], mcmc_data['C']), dtype=jnp.float64)
    mcmc_data['Y_FF'] = jnp.array(mcmc_data['Y_FF'].to_numpy().reshape(mcmc_data['A'], mcmc_data['C']), dtype=jnp.float64)
    mcmc_data['Y_MF'] = jnp.array(mcmc_data['Y_MF'].to_numpy().reshape(mcmc_data['A'], mcmc_data['C']), dtype=jnp.float64)
    mcmc_data['Y_FM'] = jnp.array(mcmc_data['Y_FM'].to_numpy().reshape(mcmc_data['A'], mcmc_data['C']), dtype=jnp.float64)
    
    return mcmc_data


def add_partsize_offsets(mcmc_data, data):
    """
    Add the participant size offsets to the data dictionary for MCMC.
    """
    unique_data = data.drop_duplicates(subset=['age', 'gender', 'part'])
    unique_data.loc[unique_data['part'] == 0, 'part'] = 1

    mcmc_data['log_N_M'] = jnp.log(unique_data[unique_data['gender'] == 'Male']['part'].to_numpy())
    mcmc_data['log_N_F'] = jnp.log(unique_data[unique_data['gender'] == 'Female']['part'].to_numpy())

    mcmc_data['log_N_M'] = jnp.tile(mcmc_data['log_N_M'], (44, 1)).T
    mcmc_data['log_N_F'] = jnp.tile(mcmc_data['log_N_F'], (44, 1)).T

    mcmc_data['log_N_M'] = jnp.array(mcmc_data['log_N_M'], dtype=jnp.float64)
    mcmc_data['log_N_F'] = jnp.array(mcmc_data['log_N_F'], dtype=jnp.float64)

    return mcmc_data



def add_pop_offsets(mcmc_data, data):
    """
    Add the population offsets to the data dictionary for MCMC.
    """

    unique_data = data[['alter_age', 'alter_gender', 'pop']].drop_duplicates()
    unique_data = unique_data.sort_values(by=['alter_age'])

    mcmc_data['P_M'] = jnp.array(unique_data[unique_data['alter_gender'] == 'Male']['pop'].to_numpy())
    mcmc_data['P_F'] = jnp.array(unique_data[unique_data['alter_gender'] == 'Female']['pop'].to_numpy())
    
    mcmc_data['log_P_M'] = jnp.log(mcmc_data['P_M'])
    mcmc_data['log_P_F'] = jnp.log(mcmc_data['P_F'])

    mcmc_data['log_P_M'] = jnp.tile(mcmc_data['log_P_M'], (44, 1))
    mcmc_data['log_P_F'] = jnp.tile(mcmc_data['log_P_F'], (44, 1))

    mcmc_data['log_P_M'] = jnp.array(mcmc_data['log_P_M'], dtype=jnp.float64)
    mcmc_data['log_P_F'] = jnp.array(mcmc_data['log_P_F'], dtype=jnp.float64)

    return mcmc_data


def add_age_strata_map(mcmc_data, strata_scheme):
    """
    Add the age strata map to the data dictionary for MCMC.
    """
    
    lookup_table = {
        "COVIMOD": ["6-9", "10-14", "15-19", "20-24", "25-34", "35-44", "45-49"],
        "CoMix": ["6-11", "12-17", "18-29", "30-39", "40-49"],
        "5yr": ["6-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
        "3yr": ["6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29",
                "30-32", "33-35", "36-38", "39-41", "42-44", "45-47", "48-49"]
    }
    

    age_strata = lookup_table[strata_scheme]
    alter_age_min = [int(re.search("^[0-9]{1,2}", stratum).group()) for stratum in age_strata]

    strata_min = list(set(alter_age_min))
    strata_min.sort()
    strata_min_idx = [x - min(strata_min) + 1 for x in strata_min]

    A = mcmc_data['A']
    C = mcmc_data['C']

    map_age_to_strata = np.zeros((A, C))

    for c in range(C):
        if c == C - 1:
            map_age_to_strata[strata_min_idx[c] - 1:A, c] = 1
        else:
            map_age_to_strata[strata_min_idx[c] - 1:strata_min_idx[c + 1] - 1, c] = 1

    mcmc_data['map_age_to_strata'] = jnp.array(map_age_to_strata)

    return mcmc_data