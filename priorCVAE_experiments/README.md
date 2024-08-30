# Model Validation

This directory provides functionality for training and evaluating PriorCVAE models.  [Weights & Biases](https://docs.wandb.ai) is used to track the performance of models and the results of experiments. [Hydra](https://hydra.cc/docs/intro/) is used for configuration.

## Installation

To install additional requirements:

```bash
pip install -r priorCVAE_experiments/requirements.txt
```

## Running an experiment

From the project's root directory run:

```bash
python -m priorCVAE_experiments.run
```

## Configuration with Hydra

Hydra configuration files can be found in `priorCVAE_experiments/conf`.


#### Overriding the default configuration

```bash
python -m priorCVAE_experiments.run kernel=Matern52
```

#### Overriding a config value

```bash
python -m model_validation.run train.num_iterations=5000 kernel.lengthscale=0.5
```