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
    'ksat_aq': 'https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/DLGXYO',
    'wte': 'https://zenodo.org/record/1315471/files/GMD-data.tar.gz?download=1'}

# Define directories:
dir = {
    'input_ksat_aq': '/home/ivanovn/ws_local/download/GLHYMPS_V2.0', 
    'input_wte': '/home/ivanovn/ws_local/download/G3M_V1.0', 
    'input_sy': '/home/ivanovn/ws_local/temp/DRYP_input', 
    'output': '/home/ivanovn/ws_local/temp/DRYP_input'}
dir['temp'] = os.path.join(dir['output'], 'temp')
# Check if the temp directory exists, if not, create it:
if not os.path.isdir(dir['temp']):
    os.mkdir(dir['temp'])

# Define the research area bounds:
bounds = {
    'min_lon': 30, 
    'max_lon': 60, 
    'min_lat': 10, 
    'max_lat': 35}

# Define the output CRS:
reproject = {
    # Lambert Aramco - EPSG:2318
    'crs_local': 'EPSG:2318', 
    # Cylindrical Equal Area - EPSG:6933
    'crs_global': 'EPSG:6933'}

# Define the output resolution [m]:
resolution = {
    'output': 1000,
    'wte': 0.01} # resolution in [°]. Needed for resampling the global water table dataset during its initial generation from a .csv

# Define the output format (options - 'GTiff', 'NetCDF', 'AAIGrid'):
saving = {
    'format': 'AAIGrid'}

# An option to plot and delete intermediate step outputs:
intermediate_step = {
    'plot': True,           # plot the intermediate step outputs
    'delete': False}        # delete the temp directory