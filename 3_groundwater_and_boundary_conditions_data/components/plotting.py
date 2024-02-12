import os
from termcolor import colored
import matplotlib.pyplot as plt
from components.timing import timer


# ------------------------------------------------------------------------------------------------
# Plotting function:
# ------------------------------------------------------------------------------------------------


# Define a function to plot the output data:
def plot_data(data, metadata, temp_path, 
              data_name=None, title=None, cbar_label=None, cmap=None,
              wgs=False):
    
    """
    Plots the output data with a colorbar, prints the information, 
    and saves the plot in the temporary directory.

    Parameters:
    ----------
    data : numpy array
        The data to be plotted.
    metadata : dict
        Dictionary containing the metadata of the output data.    
    temp_path : str
        The path to the temporary directory.
    data_name : str
        The name of the data to be plotted.
    title : str
        The title for the plot.
    cbar_label : str
        The label for the colorbar.
    cmap : str
        The colormap to be used for the plot.
    wgs : bool
        Whether the data is in WGS84 or not.
    
    Returns:
    -------
    None
    """

    # Define the extent using metadata:
    extent = (metadata['transform'][2], 
              metadata['transform'][2] + metadata['transform'][0] * metadata['width'], 
              metadata['transform'][5] + metadata['transform'][4] * metadata['height'], 
              metadata['transform'][5])
    # Create the plot:
    with timer(f"Plotting the {data_name}..."):
        fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
        # Plot the data:
        im = ax.imshow(data, cmap=cmap, extent=extent)
        # Set the title, labels, tick parameters, and grid:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        if wgs:
            ax.set_xlabel('Longitude [°]', fontsize=16, labelpad=10)
            ax.set_ylabel('Latitude [°]', fontsize=16, labelpad=10)
        else:
            ax.set_xlabel('Longitude [m]', fontsize=16, labelpad=10)
            ax.set_ylabel('Latitude [m]', fontsize=16, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        # Set the colorbar:
        colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        colorbar.ax.tick_params(labelsize=14)
        colorbar.ax.set_ylabel(cbar_label, fontsize=16, labelpad=10)
        print(colored(' ✔ Done!', 'green'))
    # Store and print the data information:
    print(colored('Data information', 'light_grey', attrs=['underline']))
    info_lines = [
        'Data size: ' + colored(f"{data.nbytes / (1024*1024*1024):.2f} GB", 'yellow'), 
        'Data dtype: ' + colored(f"{data.dtype}", 'yellow'), 
        'Data CRS: ' + colored(f"{metadata['crs']}", 'yellow'), 
        'Data shape: ' + colored(f"{data.shape}", 'yellow'), 
        'Data resolution: ' + colored(f"{metadata['transform'][0]} [m]", 'yellow')]
    print('\n'.join(info_lines))
    # Save the figure:
    with timer(f"Saving the {data_name} plot..."):
        fig.savefig(os.path.join(temp_path, f"{data_name}_plot.png"), 
                    bbox_inches='tight', format='png', dpi=300)
        print(colored(' ✔ Done!', 'green'))
    # Close the figure:
    plt.close(fig)