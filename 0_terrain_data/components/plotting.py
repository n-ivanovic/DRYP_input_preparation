# Import modules:
import os
import rasterio
import numpy as np
from termcolor import colored
from matplotlib import colors
from rasterio.plot import show
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from components.timing import timer


# ------------------------------------------------------------------------------------------------
# Plotting functions:
# ------------------------------------------------------------------------------------------------


# Define a function to plot the data tiles information:
def plot_data_tiles(data_tiles, temp_path, data_type):

    """
    Check the data extent, and provide information about the entire data list.
    
    Parameters:
    ----------
    data_tiles : list
        List of data tiles covering the specified research area.
    temp_path : str
        Full path to the temporary directory.
    data_type : str
        Type of the data (e.g. 'DEM', 'UPSTREAM').

    Returns:
    -------
    None
    """
    
    # Define a helper function to format latitude:
    def format_lat(lat):
        return f"{lat}{'°N' if lat >= 0 else '°S'}"
    # Define a helper function to format longitude:
    def format_lon(lon):
        return f"{lon}{'°E' if lon >= 0 else '°W'}"
    # Initialize variables to store information:
    with timer(f"Plotting the {data_type} extent..."):
        total_size = 0
        res_counter = Counter()
        min_lon, max_lon, min_lat, max_lat = np.inf, -np.inf, np.inf, -np.inf
        unique_shapes = set()
        crs = None
        # Initialize plot, and set title, axes labels, and grid:
        fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
        ax.set_title(f"{data_type} tiles information", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude [°]', fontsize=16)
        ax.set_ylabel('Latitude [°]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
        # Loop through all the data tiles:
        for tile in data_tiles:
            # Open the data tile:
            with rasterio.open(tile) as src:
                # Show the data data with adjusted color range:
                show(src, ax=ax, cmap='terrain')
                # Update total size:
                total_size += os.path.getsize(tile)
                # Get the bounds of the data tile and update min/max longitudes and latitudes:
                bounds = src.bounds
                min_lon = round(min(min_lon, bounds.left), 2)
                max_lon = round(max(max_lon, bounds.right), 2)
                min_lat = round(min(min_lat, bounds.bottom), 2)
                max_lat = round(max(max_lat, bounds.top), 2)
                # Increment the resolution count:
                res_counter[src.res] += 1
                # Add shape to the set of unique shapes:
                unique_shapes.add(src.shape)
                # Get coordinate reference system (assuming all tiles have the same crs):
                crs = src.crs
                # Plot the bounds of the data tile as a rectangle:
                ax.add_patch(patches.Rectangle((bounds.left, bounds.bottom), 
                                                bounds.right - bounds.left, 
                                                bounds.top - bounds.bottom, 
                                                fill=False, edgecolor='black', 
                                                linewidth=2))
        # Adjust axes:
        ax.set_xlim([min_lon, max_lon])
        ax.set_ylim([min_lat, max_lat])
        # Find the resolution:
        resolution, _ = res_counter.most_common(1)[0]
        print(colored(' ✔ Done!', 'green'))
    # Store and print the data tiles information:
    print(colored(f"{data_type} information", 'light_grey', attrs=['underline']))
    info_lines = [
        f"{data_type} tiles count: " + colored(f"{len(data_tiles)+1}", 'yellow'),
        f"{data_type} tiles size: " + colored(f"{total_size / (1024 * 1024 * 1024):.2f} GB", 'yellow'),
        f"{data_type} tiles dtype: " + colored(f"{src.dtypes[0]}", 'yellow'),
        f"{data_type} tiles CRS: " + colored(f"{crs}", 'yellow'),
        f"{data_type} tiles shape: " + colored(f"{unique_shapes}", 'yellow'),
        f"{data_type} tiles total extent: " + colored(f"LON({format_lon(min_lon)}, {format_lon(max_lon)}); LAT({format_lat(min_lat)}, {format_lat(max_lat)})", 'yellow'), 
        f"{data_type} tiles resolution: " + colored(f"{resolution}", 'yellow')]
    print('\n'.join(info_lines))
    # Save the figure:
    with timer(f"Saving the {data_type}_tiles plot..."):
        fig.savefig(os.path.join(temp_path, f"{data_type}_tiles_plot.png"), 
                    bbox_inches='tight', format='png', dpi=300)
        print(colored(' ✔ Done!', 'green'))
    # Close the figure:
    plt.close(fig)


# Define a function to plot the output data:
def plot_data(data, metadata, temp_path,
              data_name=None, title=None, cbar_label=None, cmap=None, 
              wgs=False, log_scale=False, inverse=False, binary=False):
    
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
        Whether to use labels for WGS84 coordinate reference system. Default is False.
    log_scale : bool
        Whether to use a logarithmic scale for the colorbar. Default is False.
    inverse : bool
        Whether to use a specific plot for inverted data. Default is False.
    binary : bool
        Whether to use a specific plot for binary data. Default is False.

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
        if log_scale:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=2,
                           norm=colors.LogNorm(1, data.max()), interpolation='bilinear')
        elif inverse:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=2,
                           norm=colors.Normalize(vmin=np.percentile(data, 1), 
                                                 vmax=np.percentile(data, 99)), 
                                                 interpolation='bilinear')
        else:
            im = ax.imshow(data, cmap=cmap, extent=extent, zorder=1)
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
        if binary:
            colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, 
                                    boundaries=[0, 1], values=[0.5])
            colorbar.ax.tick_params(labelsize=14)
            colorbar.ax.set_ylabel(cbar_label, fontsize=16, labelpad=10)
        else:
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