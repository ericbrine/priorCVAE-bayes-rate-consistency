import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio


def make_unique(path):

    # if not os.path.exists(path):
    #     os.makedirs(path)

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


# def plot_heatmap(matrix: jnp.ndarray, title='', output_dir='', filename='', figsize=(8,6)):
#     """ Plot a heatmap of the input matrix. """

#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#     sns.heatmap(matrix, annot=False, fmt='g', cmap='coolwarm', ax=ax)
#     ax.set_title(title)
#     plt.gca().invert_yaxis()

#     save_path = os.path.join(output_dir, filename)
#     plt.savefig(save_path)
#     return fig, ax


# plot_heatmap(y_strata.T, "Y", output_dir, filename="y_strata.png")
def plot_heatmap(matrix, title='', output_dir='', filename='', color_scale="Turbo", zmin=None, zmax=None):
    """ Plot a heatmap of the input matrix."""
    line_options = dict(width=0.5, color='white')
    fig = go.Figure(data=go.Heatmap(z=matrix, colorscale=color_scale, zmin=zmin,
        zmax=zmax, 
        xgap=0.1, 
        ygap=0.1))
    
    fig.update_layout(
        title=title, autosize=False, width=500, height=500,
        title_x=0.5,  # This centers the title
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='white',
        yaxis_gridcolor='white')

    # Save the plotly figure
    save_path = os.path.join(output_dir, filename)
    pio.write_image(fig, save_path)

    return fig
