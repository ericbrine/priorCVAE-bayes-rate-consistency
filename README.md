# Bayesian Rate Consistency Model with Deep Generative Modeling for Scalable MCMC

This repository hosts the Python, NumPyro, and JAX implementation of the Bayesian Rate Consistency Model (BRCM), enhanced with Variational Autoencoder (VAE) priors for improved scalability in Markov Chain Monte Carlo (MCMC) methods.

## PriorCVAE Package

A related repository, developed in parallel with this project and utilized within this implementation, can be accessed at:
[https://github.com/MLGlobalHealth/PriorCVAE](https://github.com/MLGlobalHealth/PriorCVAE)

## Installation

To install the necessary dependencies for this project, please execute the following command:

```bash
pip install -r requirements.txt
```

## Repository Contents

This repository is organized into two primary packages:

1. **bayes_rate_consistency** (BRCM)

   - This package contains the core implementation of the Bayesian Rate Consistency Model using Python, NumPyro, and JAX. The BRCM is the main component of the package and is designed for application to simulated datasets.
   - The `bayes_rate_consistency/config` directory employs the Hydra framework to configure model parameters.
   - The `data/simulations` directory contains simulated datasets representing both in-COVID-19 and pre-COVID-19 scenarios.
   - The `weights` directory houses pre-trained decoder models.
   - The `hpc_scripts` directory includes PBS scripts for executing experiments on the High-Performance Computing (HPC) cluster at Imperial College.

2. **priorCVAE_experiments**

   - This package is used to train and validate PriorCVAE models (Variational Autoencoders emulating Gaussian Processes). The process is managed with Hydra for configuration and Weights & Biases (WandB) for experiment tracking and logging.
   - An additional README is included within this package for further details.

## Jupyter Notebooks

The following Jupyter notebooks are included to demonstrate key functionalities:

- **GP-2D-PriorVAE.ipynb**: This notebook provides a demonstration of training two-dimensional Gaussian Processes using the PriorCVAE package.

- **PriorVAE-simulated-contact-intensity.ipynb**: This notebook illustrates the application of the Bayesian Rate Consistency Model for estimating contact intensity, leveraging the enhancements provided by PriorVAE.

