# Import modules:
import os
import rasterio
import numpy as np
import xarray as xr
from termcolor import colored
from components.timing import timer
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Saving functions:
# ------------------------------------------------------------------------------------------------

# Define a function to save the data as a GeoTIFF file:
def save_as_geotiff(data, metadata, path):
    
    """
    Save the data as a GeoTIFF file.

    Parameters:
    ----------
    data : numpy array
        The data to be saved.
    metadata : dict
        Dictionary containing the metadata of the data.
    path : str
        The path to the output file.

    Returns:
    -------
    None
    """

    with rasterio.open(path, 'w',
                       driver='GTiff',
                       height=metadata['height'],
                       width=metadata['width'],
                       count=1,
                       dtype=data.dtype,
                       crs=metadata['crs'],
                       nodata=metadata['nodata'],
                       transform=metadata['transform']) as dst:
        dst.write(data, 1)


# Define a function to save the data as a format of choice:
def save_data(data, metadata, output_path, 
              format=config.saving['format'],
              data_name=None):
    
    """
    This function saves the data as a format of choice. The supported formats are:
    - GeoTIFF ('GTiff')
    - NetCDF ('NetCDF')
    - ASCII Grid ('AAIGrid')

    Parameters:
    ----------
    data : numpy array
        The data to be saved.
    metadata : dict
        Dictionary containing the metadata of the data.
    output_path : str
        The path to the output directory.
    format : str
        Format of the output file (optional).
    data_name : str
        Name of the data to be saved (optional).

    Returns:
    -------
    None
    """

    # Save the data as a format of choice:
    with timer(f"Saving {data_name} as a {format} file..."):
        # Save as GeoTIFF:
        if format == 'GTiff':
            save_as_geotiff(data, metadata, 
                            os.path.join(output_path, f"{data_name}.tif"))
        # Save as NetCDF4:
        elif format == 'NetCDF':
            # Create x and y coordinates using metadata
            x = np.arange(metadata['width']) * metadata['transform'][0] + metadata['transform'][2]
            y = np.arange(metadata['height']) * metadata['transform'][4] + metadata['transform'][5]
            # Create a dataset:
            ds = xr.Dataset({data_name: (['y', 'x'], data)},
                             coords={'x': x, 'y': y},
                             attrs={'crs': str(metadata['crs']), 
                                    'cellsize': metadata['transform'][0]})
            # Save the dataset:
            ds.to_netcdf(os.path.join(output_path, f"{data_name}.nc"))
        # Save as ASCII Grid:
        elif format == 'AAIGrid':
            # Save as a temporary GeoTIFF:
            temp_path = os.path.join(output_path, f"{data_name}.tif")
            save_as_geotiff(data, metadata, temp_path)
            # Convert the GeoTIFF to AAIGrid using GDAL Translate:
            input_file = temp_path
            output_file = os.path.join(output_path, f"{data_name}.asc")
            command = f"gdal_translate -of AAIGrid {input_file} {output_file}"
            os.system(command)
            # Remove the temporary GeoTIFF file:
            os.remove(input_file)
        else:
            # Raise an error if the format is not supported:
            raise ValueError("Unsupported format. Choose from 'GTiff', 'NetCDF', 'AAIGrid'.")
        print(colored(' ✔ Done!', 'green'))