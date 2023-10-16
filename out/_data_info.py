# load the .nc files and plot the info in the .txt folder:
import os
import xarray as xr
import pandas as pd


# Define a function to load the .nc files and save the data info:
def load_nc():

    """
    load the .nc files and save the info about max, mean, min, std, nodata, 
    inf, size, dtype, crs, shape, extent, and cell size in a disctionary 
    for each parameter varaible, which will be saved as a .txt file with 
    the name input_data_info.txt.

    Parameters:
    ----------
    None

    Returns:
    -------
    input_data_info: dict
        a dictionary with the info about max, mean, min, std, nodata, inf, 
        size, dtype, crs, shape, extent, and resolution for each parameter 
        varaible.
    """

    # Input data folder path:
    input_path = '/home/ivanovn/ws_local/projects/ffews/data_preparation/DRYP_input/out'
    # Input data file paths:
    input_data_files = {
        # terrain data
        'cell_factor_area': os.path.join(input_path, '0_terrain_data', 'cell_factor_area.nc'),
        'chb': os.path.join(input_path, '0_terrain_data', 'chb.nc'),
        'flow_direction': os.path.join(input_path, '0_terrain_data', 'flow_direction.nc'),
        'resampled_merged_dem': os.path.join(input_path, '0_terrain_data', 'resampled_merged_dem.nc'),
        'river_lengths': os.path.join(input_path, '0_terrain_data', 'river_lengths.nc'),
        'terrain_mask': os.path.join(input_path, '0_terrain_data', 'terrain_mask.nc'),
        # soil and subsurface data
        'parameter_b': os.path.join(input_path, '2_soil_and_subsurface_data', 'parameter_b.nc'),
        'parameter_psi': os.path.join(input_path, '2_soil_and_subsurface_data', 'parameter_psi.nc'),
        'resampled_crit_wilt': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_crit_wilt.nc'),
        'resampled_ksat': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_ksat.nc'),
        'resampled_soil_depth': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_soil_depth.nc'),
        'resampled_wc_avail': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_wc_avail.nc'),
        'resampled_wc_res': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_wc_res.nc'),
        'resampled_wc_sat': os.path.join(input_path, '2_soil_and_subsurface_data', 'resampled_wc_sat.nc'),
        'sigma_ksat': os.path.join(input_path, '2_soil_and_subsurface_data', 'sigma_ksat.nc'),
        # groundwater and boundary conditions data
        'wte': os.path.join(input_path, '3_groundwater_and_boundary_conditions_data', 'wte.nc'),
        'ksat_aq': os.path.join(input_path, '3_groundwater_and_boundary_conditions_data', 'ksat_aq.nc')}
    # Create a dictionary to save the info about the input data:
    input_data_info = {}
    # Create an empty DataFrame to store the tabulated data
    df_list = []
    for key, value in input_data_files.items():
        with xr.open_dataset(value) as ds:
            var = ds[key]
            new_row = {
                'Variable': key,
                'Max': float(var.where(var != -9999).max().values),
                'Mean': float(var.where(var != -9999).mean().values),
                'Min': float(var.where(var != -9999).min().values), 
                'Std': float(var.std().values),
                'Size': f"{var.nbytes / (1024*1024*1024):.2f} GB", 
                'dtype': str(var.dtype),
                'crs': 'EPSG:4326',
                'Shape': str(var.shape),
                'Cell_Size': '0.01'}
            df_list.append(pd.DataFrame([new_row]))
            input_data_info[key] = new_row
    # Concatenate the DataFrames
    df = pd.concat(df_list, ignore_index=True)
    # Save DataFrame to a .csv file with comma as a separator
    csv_file_path = os.path.join(input_path, '_data_info.csv')
    df.to_csv(csv_file_path, index=False)
    # Return the dictionary:
    return input_data_info

# Call the function
input_data_info = load_nc()