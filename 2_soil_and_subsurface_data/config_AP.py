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
    'layers': 'https://www.dropbox.com/sh/iaj2ika2t1pr7lr/AADpmMOal6VXModX5HL4aooFa/Layers?dl=0&lst=',
    'top_subsoil': 'https://www.dropbox.com/sh/iaj2ika2t1pr7lr/AAAoXQ-bnh-oAodIlra_hUcya/Top_Subsoil?dl=0&lst=',
    'soil_depth': 'https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif'}

# Define directories:
dir = {
    'input': '/home/ivanovn/ws_local/download/HiHydroSoil_v2.0',
    'output': '/home/ivanovn/ws_local/projects/ffews/data_processing_DRYP/out'}
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
    'output': 1000}

# Define the output format (options - 'GTiff', 'NetCDF', 'AAIGrid'):
saving = {
    'format': 'AAIGrid'}

# An option to plot and delete intermediate step outputs:
intermediate_step = {
    'plot': True,           # plot the intermediate step outputs
    'delete': False}        # delete the temp directory