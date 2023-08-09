import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


def make_unique(path):

    # if not os.path.exists(path):
    #     os.makedirs(path)

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def plot_heatmap(matrix: jnp.ndarray, title='', output_dir='', filename='', figsize=(8,6)):
    """ Plot a heatmap of the input matrix. """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.heatmap(matrix, annot=False, fmt='g', cmap='coolwarm', ax=ax)
    ax.set_title(title)
    plt.gca().invert_yaxis()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    return fig, ax