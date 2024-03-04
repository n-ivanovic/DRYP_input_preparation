"""
This is a configuration file for the Arabian Peninsula (AP) region.
------------------------------------------------------------------
If you wish to apply the configuration file to another region, please change the 
values of the parameters in this file accordingly. Afterward, save the file under 
a new name, and load the new configuration file in the config loader function (see 
the config loader function in the config_loader.py file).
"""

# Import modules:
import os


# URL to the data source:
url = {
    'DEM': 'https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/',
    'UPSTREAM': 'https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/'}

# Define the authentication credentials:
auth = {
    'DEM': ('globaldem', 'preciseelevation'), 
    'UPSTREAM': ('hydrography', 'rivernetwork')}

# Define directories:
dir = {
    'input_dem': '/home/ivanovn/ws_local/download/MERIT/DEM',
    'input_upstream': '/home/ivanovn/ws_local/download/MERIT/UPSTREAM',
    'output': '/home/ivanovn/ws_local/temp/DRYP_input/baysh'}
dir['temp'] = os.path.join(dir['output'], 'temp')
# Check if the temp directory exists, if not, create it:
if not os.path.isdir(dir['temp']):
    os.mkdir(dir['temp'])

# Define the research area bounds:
bounds = {
    'min_lon': 42.3,          # 1 degree less to avoid adding the next tile
    'max_lon': 43.6,          # 1 degree less to avoid adding the next tile
    'min_lat': 17, 
    'max_lat': 18.2}

# Define the terrain mask polygon bounds:
polygon = {
    'bounds': '/home/ivanovn/ws_local/temp/DRYP_input/baysh/baysh.shp'}

# Define the output CRS:
reproject = {
    'crs': 'EPSG:6933'}     # Cylindrical Equal Area - EPSG:6933

# Define the output resolution for projected coordinate system (meters):
resolution = {
    'output': 1000}

# Define the threshold areal coverage for the river network extraction [m^2]:
threshold = {
    'area': 1e8}

# Define the output format (options - 'GTiff', 'NetCDF', 'AAIGrid'):
saving = {
    'format': 'AAIGrid'}

# An option to plot and delete intermediate step outputs:
intermediate_step = {
    'plot': True,           # plot the intermediate step outputs
    'delete': False}        # delete the temp directory