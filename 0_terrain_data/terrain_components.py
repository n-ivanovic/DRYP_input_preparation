##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "August 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__department__  = "Earth Science and Engineering"
__institution__ = "King Abdullah University of Science and Technology, SA"

##################################################################################################
######################################### Headline ###############################################
##################################################################################################

"""
Aim:
---
This script prepares terrain data for the DRYP 1.0 model. The script downloads the data, defines
the research area extent, merges the data, resamples the data, and calculates terrain parameters.

Input:
-----
1. DEM tiles (downloaded from MERIT Hydro website);
2. UPSTREAM tiles (downloaded from MERIT Hydro website).

Operations:
----------
1. Download the data;
2. Define the research area extent;
3. Merge the data;
4. Resample the data;
5. Calculate terrain parameters.

Outputs:
-------
1. Topography (DEM)			                    -> 	           	  res_merged_dem
2. Cell factor area     		                ->	    	      cell_area
3. Flow direction				                ->	    	      flow_dir
4. Boundary conditions (CHB)                    ->	              chb
5. Basin mask     	                            ->	              terrain_mask
6. River lenghts				                ->	              river_lengths
7. River widths				                    ->	              None
8. River bottom elevation                       ->	              None
"""

##################################################################################################
###################################  Main body of the script  ####################################
##################################################################################################

# Import modules:
import os
from components.timing import script_timer
from components.configuration import config
from components.downloading import download_data
from components.gathering import get_data
from components.merging import merge_data
from components.reprojecting import resample_data
from components.parameters import (cell_factor_area, invert_upstream, flow_accumulation,
                                   river_network, boundary_conditions, terrain_mask, river_lengths)

##################################################################################################
# Script timer:
##################################################################################################

# Display the runtime of the script at the end:
runtime = script_timer()

##################################################################################################
# Set the directories:
##################################################################################################

# Get the directories:
input_dir_ups = config.dir['input_upstream']
input_dir_dem = config.dir['input_dem']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# # Download the DEM tiles (if not already downloaded):
# download_data(input_dir_dem, 'DEM')
# # Download the upstream data (if not already downloaded):
# download_data(input_dir_ups, 'UPSTREAM')

##################################################################################################
# Define the research area extent:
##################################################################################################

# Get DEM and UPSTREAM tiles:
dem_tiles, ups_tiles = get_data(input_dir_dem, input_dir_ups, temp_dir)

##################################################################################################
# Merge the data:
##################################################################################################

# Merge DEM and UPSTREAM tiles:
merged_dem, merged_ups, metadata = merge_data(dem_tiles, ups_tiles, temp_dir)

##################################################################################################
# Resample the data:
##################################################################################################

# Resample merged DEM and UPSTREAM data:
res_merged_dem, res_merged_ups, res_metadata = resample_data(merged_dem, merged_ups, metadata, 
                                                             temp_dir, output_dir)

##################################################################################################
# Calculate terrain parameters:
##################################################################################################

# Create cell factor area:
cell_area = cell_factor_area(res_merged_dem, res_metadata, output_dir)
# Invert merged upstream data:
inv_res_ups = invert_upstream(res_merged_ups, res_metadata, temp_dir, output_dir)
# Compute flow direction and flow accumulation:
flow_dir, flow_acc = flow_accumulation(inv_res_ups, res_metadata, temp_dir, output_dir)
# Extract river network based on the speficied areal coverage threshold:
riv_net = river_network(flow_acc, cell_area, res_metadata, temp_dir)
# Compute constant head boundary conditions (CHB):
chb = boundary_conditions(riv_net, res_merged_dem, res_metadata, temp_dir, output_dir)
# Create research area terrain mask:
terr_mask = terrain_mask(merged_dem, metadata, temp_dir, output_dir)
# Compute river lengths:
riv_len = river_lengths(riv_net, res_metadata, temp_dir, output_dir)
 
##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.system(f"rm -r {temp_dir}")

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################